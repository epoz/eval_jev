"""
Rank Iconclass candidates per caption using the Claude API (§4.3 in Plan.md).

Unlike classify_captions_claude.py (which asks for an independent per-code
probability - a calibration Claude has no real basis for), this asks for a
single ranking judgment per caption: order all candidate codes from most to
least likely to apply. Closer to what an LLM naturally does, and directly
comparable to Jev's probability-sorted order via rank-correlation / top-K
overlap metrics (see Plan.md §4.1).

Design notes
------------
* One API call per caption, no chunking. Call-count parity with Jev
  (classify_captions_claude.py's reason for chunking at 30) doesn't matter
  here - only the output ranking does. This is also the main token-saving
  lever: no repeated system-prompt/caption overhead for the ~25 items that
  needed 2-3 chunks under the chunked script.
* Response schema is an ordered array of code strings only - no labels, no
  probability floats. Claude already has the labels in its input; paying to
  echo them back (or fabricate a probability) buys nothing. Labels are
  reattached locally, for free, after the call.
* If Claude drops or invents a code, missing candidates are backfilled at
  the end of the ranking (worst rank) so every candidate still gets a rank
  for downstream comparison; unknown codes are logged and dropped.

Run
---
    export ANTHROPIC_API_KEY=...
    python rank_captions_claude.py --limit 3      # sanity check
    python rank_captions_claude.py                # bulk run (resumes)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, Field, TypeAdapter

log = logging.getLogger("rank_captions_claude")

MODEL = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Typed input / output models
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    code: str
    label: str


class InputItem(BaseModel):
    s: int
    caption: str
    IC: list[Candidate]


class RankedCode(BaseModel):
    code: str
    label: str
    rank: int  # 1 = most likely to apply


class OutputItem(BaseModel):
    s: int
    caption: str
    elapsed_seconds: float = Field(ge=0.0)
    ranking: list[RankedCode]


class RankingResponse(BaseModel):
    """Claude's raw structured-output shape: an ordered array of code
    strings, most to least likely. Labels are reattached locally."""

    ranking: list[str]


# ---------------------------------------------------------------------------
# Input parsing - same shape as classify_captions.py / classify_captions_claude.py
# ---------------------------------------------------------------------------


def _coerce_input(raw: list[dict[str, Any]]) -> list[InputItem]:
    items: list[InputItem] = []
    for row in raw:
        ic_pairs = row.get("IC", [])
        candidates = [Candidate(code=code, label=label) for code, label in ic_pairs]
        items.append(InputItem(s=row["s"], caption=row["caption"], IC=candidates))
    return items


def load_input(path: Path) -> list[InputItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected top-level JSON array in {path}, got {type(raw)}")
    return _coerce_input(raw)


# ---------------------------------------------------------------------------
# Prompt building - compact encoding (no indent) to save input tokens
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Iconclass cataloguer. Given a caption and a list of "
    "candidate Iconclass codes, rank ALL candidates from most to least "
    "likely to apply to the caption. Return every candidate code exactly "
    "once, in ranked order, using the code strings given."
)


def _user_prompt(caption: str, candidates: list[Candidate]) -> str:
    lines = "\n".join(f"{c.code}\t{c.label}" for c in candidates)
    return f"Caption:\n{caption}\n\nCandidates ({len(candidates)}):\n{lines}"


# ---------------------------------------------------------------------------
# Per-item ranking
# ---------------------------------------------------------------------------


async def rank_item(
    client: anthropic.AsyncAnthropic,
    item: InputItem,
    *,
    model: str = MODEL,
    max_tokens: int = 1024,
) -> OutputItem:
    labels_by_code = {c.code: c.label for c in item.IC}
    all_codes = list(labels_by_code.keys())

    start = time.perf_counter()
    response = await client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(item.caption, item.IC)}],
        output_format=RankingResponse,
    )
    elapsed = time.perf_counter() - start

    ranking = response.parsed_output.ranking
    seen: set[str] = set()
    ordered: list[str] = []
    unknown: list[str] = []
    for code in ranking:
        if code in seen:
            continue
        seen.add(code)
        if code in labels_by_code:
            ordered.append(code)
        else:
            unknown.append(code)

    missing = [c for c in all_codes if c not in seen]
    if missing:
        log.warning("s=%s missing from ranking, appended at end: %s", item.s, missing)
        ordered.extend(missing)
    if unknown:
        log.warning("s=%s ranked unknown codes (dropped): %s", item.s, unknown)

    ranked = [
        RankedCode(code=code, label=labels_by_code[code], rank=i + 1)
        for i, code in enumerate(ordered)
    ]
    return OutputItem(s=item.s, caption=item.caption, elapsed_seconds=elapsed, ranking=ranked)


# ---------------------------------------------------------------------------
# Driver - same (s, caption)-keyed resume/redo logic as classify_captions_claude.py
# ---------------------------------------------------------------------------


def _key(s: int, caption: str) -> tuple[int, str]:
    return (s, caption)


def _load_existing(output_path: Path) -> dict[tuple[int, str], OutputItem]:
    if not output_path.exists():
        return {}
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    adapter = TypeAdapter(list[OutputItem])
    return {_key(item.s, item.caption): item for item in adapter.validate_python(raw)}


async def run(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    concurrency: int,
    limit: int | None,
    redo: bool,
    max_tokens: int,
) -> None:
    items = load_input(input_path)
    existing = {} if redo else _load_existing(output_path)
    todo = [it for it in items if _key(it.s, it.caption) not in existing]
    if limit is not None:
        todo = todo[:limit]
    log.info(
        "Loaded %d captions from %s (%d already in %s, %d to rank)",
        len(items), input_path, len(existing), output_path, len(todo),
    )

    sem = asyncio.Semaphore(concurrency)
    client = anthropic.AsyncAnthropic()

    async def _one(it: InputItem) -> OutputItem:
        async with sem:
            try:
                return await rank_item(client, it, model=model, max_tokens=max_tokens)
            except Exception:
                log.exception("Ranking failed for s=%s", it.s)
                raise

    total_start = time.perf_counter()
    new_results = await asyncio.gather(*(_one(it) for it in todo))
    total_seconds = time.perf_counter() - total_start

    by_key = {**existing, **{_key(r.s, r.caption): r for r in new_results}}
    results = [
        by_key[_key(it.s, it.caption)] for it in items if _key(it.s, it.caption) in by_key
    ]

    adapter = TypeAdapter(list[OutputItem])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(adapter.dump_python(results, mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Wrote %d rankings to %s (%d new, %.2fs total)",
        len(results), output_path, len(new_results), total_seconds,
    )
    if new_results:
        elapsed = [r.elapsed_seconds for r in new_results]
        log.info(
            "per-caption API: min=%.3fs mean=%.3fs max=%.3fs  api calls=%d",
            min(elapsed), sum(elapsed) / len(elapsed), max(elapsed), len(new_results),
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).parent.parent
    p.add_argument("--input", type=Path, default=project_dir / "data" / "candidates.json")
    p.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "output" / "rankings_claude.json",
    )
    p.add_argument("--model", type=str, default=MODEL)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N captions not already in --output.",
    )
    p.add_argument(
        "--redo",
        action="store_true",
        help="Rerank everything, ignoring any existing --output results.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Response token ceiling. Bump for models with default-on "
        "thinking (e.g. Opus 5) - thinking tokens count against this cap "
        "and can truncate the structured output at 1024.",
    )
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(
        run(
            args.input,
            args.output,
            model=args.model,
            concurrency=args.concurrency,
            limit=args.limit,
            redo=args.redo,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
