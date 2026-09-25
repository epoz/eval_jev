"""
Multi-label Iconclass classification for captions using the Claude API.

Same task as ``classify_captions.py`` (Jev), for comparison: for every caption
in ``candidates.json``, score how likely each candidate Iconclass concept is
depicted / referenced by the caption. Output shape matches the Jev script's
``OutputItem``/``Prediction`` models so the two prediction files can be
compared directly.

Design notes
------------
* Chunked the same way as ``classify_captions.py``: candidate lists are split
  into groups of ``MAX_CANDIDATES_PER_CALL`` (30, matching Jev's
  ``MAX_NOULS_PER_CALL``), one API call per chunk, so call counts and timing
  are directly comparable rather than Claude doing the whole candidate list
  in a single call. At chunk size 30 this produces 123 total calls across the
  97-item dataset - the same total Jev's script reports.
* Each chunk is one structured-output call asking for a typed
  ``{code, probability}`` array back. ``output_format`` (Pydantic) guarantees
  valid, schema-matching JSON - no manual parsing/repair needed.
* ``certainty`` is computed the same way as the Jev script
  (``abs(probability - 0.5) * 2``) purely so downstream comparison code can
  treat both prediction files identically. It does not imply Claude's
  probabilities are calibrated the same way Jev's Noul is.
* ``timing`` mirrors Jev's ``Timing`` model (``n_calls``, ``n_candidates``,
  ``api_seconds``, ``wall_seconds``) field-for-field, and the end-of-run
  report format matches Jev's ``_report_timing`` output line-for-line.

Run
---
    export ANTHROPIC_API_KEY=...
    pip install anthropic pydantic
    python classify_captions_claude.py \
        --input  candidates.json \
        --output predictions_claude.json \
        --concurrency 4 \
        --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, Field, TypeAdapter

log = logging.getLogger("classify_captions_claude")

# Haiku 4.5 - cheapest current model ($1.00/$5.00 per 1M vs Opus 5's
# $5.00/$25.00) - picked deliberately here for budget reasons. This is a
# case-by-case cost/quality tradeoff, not a general default.
MODEL = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Typed input / output models (mirrors classify_captions.py)
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    code: str
    label: str


class InputItem(BaseModel):
    s: int
    caption: str
    IC: list[Candidate]


class Prediction(BaseModel):
    code: str
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    certainty: float = Field(ge=0.0, le=1.0)


class Timing(BaseModel):
    """Wall-clock timing for one caption's classification.

    Mirrors classify_captions.py's ``Timing`` model field-for-field.
    ``api_seconds`` is the sum of the awaited API calls for this caption
    (one per chunk); ``wall_seconds`` includes local overhead (request
    building, JSON parsing); ``n_calls`` is the number of API round-trips,
    ``n_candidates`` the total candidates scored across those calls.
    """

    n_calls: int
    n_candidates: int
    api_seconds: float
    wall_seconds: float


class OutputItem(BaseModel):
    s: int
    caption: str
    predictions: list[Prediction]
    timing: Timing


# ---------------------------------------------------------------------------
# Claude's structured-output response shape: one probability per candidate
# code, matched back to labels locally after the call.
# ---------------------------------------------------------------------------


class ScoredCode(BaseModel):
    code: str
    probability: float = Field(ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    predictions: list[ScoredCode]


# ---------------------------------------------------------------------------
# Input parsing - same flat-array-with-[code, label]-pairs shape as
# classify_captions.py.
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
# Prompt building
# ---------------------------------------------------------------------------

MAX_CANDIDATES_PER_CALL = 30
"""Upper bound on candidates scored per API call - matches Jev's
MAX_NOULS_PER_CALL so call counts and per-call timing are comparable."""

SYSTEM_PROMPT = (
    "You are an expert Iconclass cataloguer. For an image caption and a list "
    "of candidate Iconclass concepts, score how likely each concept is "
    "depicted, contained, or otherwise matched by the caption. Score every "
    "candidate independently as a probability in [0, 1] - do not assume the "
    "scores across candidates sum to 1. Return every candidate code exactly "
    "once, using the code strings given."
)


def _user_prompt(caption: str, candidates: list[Candidate]) -> str:
    cand_dicts = [{"code": c.code, "label": c.label} for c in candidates]
    return (
        f"Caption:\n{caption}\n\n"
        f"Candidate Iconclass concepts ({len(cand_dicts)}):\n"
        f"{json.dumps(cand_dicts, ensure_ascii=False, indent=2)}"
    )


def _chunk(xs: list, size: int) -> list[list]:
    return [xs[i : i + size] for i in range(0, len(xs), size)]


# ---------------------------------------------------------------------------
# Per-item classification
# ---------------------------------------------------------------------------


async def classify_item(
    client: anthropic.AsyncAnthropic,
    item: InputItem,
    *,
    model: str = MODEL,
    max_candidates_per_call: int = MAX_CANDIDATES_PER_CALL,
) -> OutputItem:
    labels_by_code = {c.code: c.label for c in item.IC}
    predictions: list[Prediction] = []

    n_calls = 0
    n_candidates = 0
    api_seconds = 0.0
    wall_start = time.perf_counter()

    for chunk_idx, chunk in enumerate(_chunk(item.IC, max_candidates_per_call)):
        call_start = time.perf_counter()
        response = await client.messages.parse(
            model=model,
            max_tokens=4096,  # sized for <=30 short {code, probability} objects
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(item.caption, chunk)}],
            output_format=ClassificationResponse,
        )
        call_seconds = time.perf_counter() - call_start

        api_seconds += call_seconds
        n_calls += 1
        n_candidates += len(chunk)
        log.info(
            "s=%s chunk=%d candidates=%d api=%.3fs (%.1f candidates/s)",
            item.s,
            chunk_idx,
            len(chunk),
            call_seconds,
            len(chunk) / call_seconds if call_seconds > 0 else float("inf"),
        )

        chunk_codes = {c.code for c in chunk}
        scored_codes = {sc.code for sc in response.parsed_output.predictions}
        missing = chunk_codes - scored_codes
        extra = scored_codes - chunk_codes
        if missing:
            log.warning(
                "s=%s chunk=%d missing scores for codes: %s",
                item.s, chunk_idx, sorted(missing),
            )
        if extra:
            log.warning(
                "s=%s chunk=%d scored unknown codes (dropped): %s",
                item.s, chunk_idx, sorted(extra),
            )

        predictions.extend(
            Prediction(
                code=sc.code,
                label=labels_by_code[sc.code],
                probability=sc.probability,
                certainty=abs(sc.probability - 0.5) * 2.0,
            )
            for sc in response.parsed_output.predictions
            if sc.code in labels_by_code
        )

    predictions.sort(key=lambda p: p.probability, reverse=True)
    wall_seconds = time.perf_counter() - wall_start
    return OutputItem(
        s=item.s,
        caption=item.caption,
        predictions=predictions,
        timing=Timing(
            n_calls=n_calls,
            n_candidates=n_candidates,
            api_seconds=api_seconds,
            wall_seconds=wall_seconds,
        ),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


# candidates.json has been observed to contain duplicate `s` values covering
# distinct captions - key on (s, caption) everywhere below, not `s` alone, or
# duplicates silently collide and one caption's result overwrites the other's.


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
    max_candidates_per_call: int,
) -> None:
    items = load_input(input_path)
    existing = {} if redo else _load_existing(output_path)
    todo = [it for it in items if _key(it.s, it.caption) not in existing]
    if limit is not None:
        todo = todo[:limit]
    log.info(
        "Loaded %d captions from %s (%d already in %s, %d to classify)",
        len(items), input_path, len(existing), output_path, len(todo),
    )

    sem = asyncio.Semaphore(concurrency)
    client = anthropic.AsyncAnthropic()

    async def _one(it: InputItem) -> OutputItem:
        async with sem:
            try:
                return await classify_item(
                    client, it, model=model, max_candidates_per_call=max_candidates_per_call
                )
            except Exception:
                log.exception("Classification failed for s=%s", it.s)
                raise

    total_start = time.perf_counter()
    new_results = await asyncio.gather(*(_one(it) for it in todo))
    total_seconds = time.perf_counter() - total_start

    by_key = {
        **existing,
        **{_key(r.s, r.caption): r for r in new_results},
    }
    # Preserve candidates.json order for every item classified so far.
    results = [
        by_key[_key(it.s, it.caption)]
        for it in items
        if _key(it.s, it.caption) in by_key
    ]

    adapter = TypeAdapter(list[OutputItem])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(adapter.dump_python(results, mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Wrote %d predictions to %s (%d new)", len(results), output_path, len(new_results)
    )

    if new_results:
        _report_timing(new_results, total_seconds=total_seconds, concurrency=concurrency)
    else:
        log.info("Nothing new classified this run - skipping timing report.")


def _report_timing(
    results: list[OutputItem], *, total_seconds: float, concurrency: int
) -> None:
    """Print an aggregate timing report in the same format as
    classify_captions.py's _report_timing, for direct comparison.

    Covers only the items classified in *this* run (``new_results``), since
    a resumed run's ``total_seconds`` wouldn't otherwise correspond to the
    items it actually processed.
    """
    api = [r.timing.api_seconds for r in results]
    wall = [r.timing.wall_seconds for r in results]
    n_candidates = sum(r.timing.n_candidates for r in results)
    n_calls = sum(r.timing.n_calls for r in results)

    def _stats(xs: list[float]) -> str:
        return (
            f"min={min(xs):.3f}s  median={statistics.median(xs):.3f}s  "
            f"mean={statistics.fmean(xs):.3f}s  max={max(xs):.3f}s"
        )

    log.info("── timing report ──")
    log.info("captions        : %d", len(results))
    log.info("api calls       : %d", n_calls)
    log.info("candidates      : %d", n_candidates)
    log.info("concurrency     : %d", concurrency)
    log.info("per-caption API : %s", _stats(api))
    log.info("per-caption wall: %s", _stats(wall))
    log.info(
        "per-call API    : mean=%.3fs  (%.1f candidates/s per call)",
        sum(api) / n_calls if n_calls else 0.0,
        n_candidates / sum(api) if sum(api) > 0 else float("inf"),
    )
    log.info(
        "total wall time : %.2fs  (%.2f captions/s, %.1f candidates/s end-to-end)",
        total_seconds,
        len(results) / total_seconds if total_seconds > 0 else float("inf"),
        n_candidates / total_seconds if total_seconds > 0 else float("inf"),
    )
    log.info(
        "speedup vs serial: %.2fx  (sum(api)=%.1fs / wall=%.1fs)",
        sum(api) / total_seconds if total_seconds > 0 else float("inf"),
        sum(api),
        total_seconds,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # candidates.json lives in data/, one directory up from scripts/
    project_dir = Path(__file__).parent.parent
    p.add_argument("--input", type=Path, default=project_dir / "data" / "candidates.json")
    p.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "output" / "predictions_claude.json",
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
        help="Reclassify everything, ignoring any existing --output results.",
    )
    p.add_argument(
        "--max-candidates-per-call",
        type=int,
        default=MAX_CANDIDATES_PER_CALL,
        help="Split a candidate list into chunks of this many per API call "
        "(matches Jev's --max-nouls-per-call for comparability).",
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
            max_candidates_per_call=args.max_candidates_per_call,
        )
    )


if __name__ == "__main__":
    main()
