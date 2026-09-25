"""
Multi-choice Iconclass classification for captions using TypeSafe's Jev model.

For every caption in ``candidates.json``, we ask Jev — via a fan-out of Noul
(yes/no) questions, one per candidate Iconclass concept — how likely each
candidate concept is depicted / referenced by the caption. The result is a
ranked, calibrated list of Iconclass candidates per caption.

Design notes
------------
* Each item in ``candidates.json`` looks like::

      {"s": <int>, "caption": <str>, "IC": [[<code>, <label>], ...]}

  The candidate lists have up to ~60 entries. We treat this as a *multi-label*
  problem (several Iconclass concepts can apply to one image / caption),
  which maps cleanly onto Jev's Noul primitive: one Noul per candidate,
  answered in a single ``system_one`` call (speculative fan-out pattern).

* Iconclass codes contain characters that are not safe as JSON keys (``(``,
  ``+``, ``)``). We therefore use synthetic keys ``q000``, ``q001``, ...
  in the request and keep a mapping back to the real ``(code, label)`` tuple.

* If a candidate list exceeds ``MAX_NOULS_PER_CALL``, we split it into
  chunks and issue one ``system_one`` call per chunk (still one call per
  chunk, questions inside the chunk are answered in parallel by Jev).

Run
---
    export TYPESAFE_API_KEY=...
    uv add typesafe-sdk pydantic       # or: pip install typesafe-sdk pydantic
    python classify_captions.py \
        --input  candidates.json \
        --output predictions.json \
        --concurrency 4
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

from pydantic import BaseModel, Field, TypeAdapter
from typesafe_sdk import AsyncTypeSafeClient, Noul

log = logging.getLogger("classify_captions")

# ---------------------------------------------------------------------------
# Typed input / output models
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """A single Iconclass candidate for a caption."""

    code: str
    label: str


class InputItem(BaseModel):
    """One row of the input file."""

    s: int
    caption: str
    IC: list[Candidate]


class Prediction(BaseModel):
    """Per-candidate model output.

    ``NoulAnswer`` only exposes ``noul`` (the probability of yes); the Noul
    primitive intentionally has no separate ``confidence`` field — |noul-0.5|
    already encodes certainty. We surface that as ``certainty`` for
    convenience so callers can threshold on it directly.
    """

    code: str
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    certainty: float = Field(ge=0.0, le=1.0)


class Timing(BaseModel):
    """Wall-clock timing for one caption's classification.

    ``api_seconds`` is the sum of the awaited ``system_one`` calls for this
    caption (one per chunk). ``wall_seconds`` is elapsed time including any
    local overhead (request building, JSON parsing). ``n_calls`` is the
    number of ``system_one`` round-trips, ``n_nouls`` the total number of
    Noul questions asked across those calls.
    """

    n_calls: int
    n_nouls: int
    api_seconds: float
    wall_seconds: float


class OutputItem(BaseModel):
    """Ranked predictions for one caption."""

    s: int
    caption: str
    predictions: list[Prediction]
    timing: Timing


# ---------------------------------------------------------------------------
# Input parsing — the file is a flat JSON array of InputItem objects, but the
# inner "IC" field is a list of [code, label] pairs, not objects. We validate
# via a custom pre-parser.
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
# Question building
# ---------------------------------------------------------------------------

MAX_NOULS_PER_CALL = 30
"""Upper bound on Nouls per system_one call. Splitting keeps individual
requests small, which helps latency and stays clear of per-request limits."""


def _noul_instructions(candidate: Candidate) -> str:
    """Human-readable yes/no criterion for one candidate.

    Jev's Noul primitive returns P(yes | state, instructions). We phrase the
    question as a positive assertion so ``noul`` is the probability that the
    Iconclass concept is depicted / referenced by the caption.
    """
    return (
        "The image described by the caption depicts, contains, or otherwise "
        f"matches the following Iconclass concept:\n"
        f"  code:  {candidate.code}\n"
        f"  label: {candidate.label}"
    )


def _chunk[T](xs: list[T], size: int) -> list[list[T]]:
    return [xs[i : i + size] for i in range(0, len(xs), size)]


# ---------------------------------------------------------------------------
# Per-item classification
# ---------------------------------------------------------------------------


async def classify_item(
    client: AsyncTypeSafeClient,
    item: InputItem,
    *,
    max_nouls_per_call: int = MAX_NOULS_PER_CALL,
) -> OutputItem:
    """Ask Jev one Noul per Iconclass candidate; return ranked predictions."""

    state = {"caption": item.caption}
    predictions: list[Prediction] = []

    n_calls = 0
    n_nouls = 0
    api_seconds = 0.0
    wall_start = time.perf_counter()

    for chunk_idx, chunk in enumerate(_chunk(item.IC, max_nouls_per_call)):
        # Build synthetic key -> candidate map for this chunk.
        keys: dict[str, Candidate] = {
            f"q{chunk_idx:02d}_{i:03d}": cand for i, cand in enumerate(chunk)
        }
        questions = {
            key: Noul(instructions=_noul_instructions(cand))
            for key, cand in keys.items()
        }
        log.debug("s=%s chunk=%d nouls=%d", item.s, chunk_idx, len(questions))

        call_start = time.perf_counter()
        result = await client.system_one(state=state, questions=questions)
        call_seconds = time.perf_counter() - call_start

        api_seconds += call_seconds
        n_calls += 1
        n_nouls += len(questions)
        log.info(
            "s=%s chunk=%d nouls=%d api=%.3fs (%.1f nouls/s)",
            item.s,
            chunk_idx,
            len(questions),
            call_seconds,
            len(questions) / call_seconds if call_seconds > 0 else float("inf"),
        )

        for key, cand in keys.items():
            answer = result.nouls[key]
            p = float(answer.noul)
            predictions.append(
                Prediction(
                    code=cand.code,
                    label=cand.label,
                    probability=p,
                    certainty=abs(p - 0.5) * 2.0,
                )
            )

    predictions.sort(key=lambda p: p.probability, reverse=True)
    wall_seconds = time.perf_counter() - wall_start
    return OutputItem(
        s=item.s,
        caption=item.caption,
        predictions=predictions,
        timing=Timing(
            n_calls=n_calls,
            n_nouls=n_nouls,
            api_seconds=api_seconds,
            wall_seconds=wall_seconds,
        ),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run(
    input_path: Path,
    output_path: Path,
    *,
    concurrency: int,
    limit: int | None,
    max_nouls_per_call: int,
) -> None:
    items = load_input(input_path)
    if limit is not None:
        items = items[:limit]
    log.info("Loaded %d captions from %s", len(items), input_path)

    sem = asyncio.Semaphore(concurrency)

    total_start = time.perf_counter()
    async with AsyncTypeSafeClient() as client:

        async def _one(it: InputItem) -> OutputItem:
            async with sem:
                try:
                    return await classify_item(
                        client, it, max_nouls_per_call=max_nouls_per_call
                    )
                except Exception:
                    log.exception("Classification failed for s=%s", it.s)
                    raise

        results = await asyncio.gather(*(_one(it) for it in items))
    total_seconds = time.perf_counter() - total_start

    adapter = TypeAdapter(list[OutputItem])
    output_path.write_text(
        json.dumps(adapter.dump_python(results, mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Wrote %d predictions to %s", len(results), output_path)

    _report_timing(results, total_seconds=total_seconds, concurrency=concurrency)


def _report_timing(
    results: list[OutputItem], *, total_seconds: float, concurrency: int
) -> None:
    """Print an aggregate timing report suitable for LLM-vs-Jev comparison.

    Two views matter when comparing to an LLM baseline:

    * **Per-caption wall time** — how long a user waits for one prediction.
      With Jev this is roughly ``api_seconds`` for the caption; with a chat
      LLM baseline you would measure the same thing.
    * **Throughput** — total wall time divided by number of captions (or
      Nouls), reflecting how the concurrency setting scales.
    """
    if not results:
        log.info("No results — nothing to time.")
        return

    api = [r.timing.api_seconds for r in results]
    wall = [r.timing.wall_seconds for r in results]
    n_nouls = sum(r.timing.n_nouls for r in results)
    n_calls = sum(r.timing.n_calls for r in results)

    def _stats(xs: list[float]) -> str:
        return (
            f"min={min(xs):.3f}s  median={statistics.median(xs):.3f}s  "
            f"mean={statistics.fmean(xs):.3f}s  max={max(xs):.3f}s"
        )

    log.info("── timing report ──")
    log.info("captions        : %d", len(results))
    log.info("system_one calls: %d", n_calls)
    log.info("noul questions  : %d", n_nouls)
    log.info("concurrency     : %d", concurrency)
    log.info("per-caption API : %s", _stats(api))
    log.info("per-caption wall: %s", _stats(wall))
    log.info(
        "per-call API    : mean=%.3fs  (%.1f nouls/s per call)",
        sum(api) / n_calls if n_calls else 0.0,
        n_nouls / sum(api) if sum(api) > 0 else float("inf"),
    )
    log.info(
        "total wall time : %.2fs  (%.2f captions/s, %.1f nouls/s end-to-end)",
        total_seconds,
        len(results) / total_seconds if total_seconds > 0 else float("inf"),
        n_nouls / total_seconds if total_seconds > 0 else float("inf"),
    )
    log.info(
        "speedup vs serial: %.2fx  (sum(api)=%.1fs / wall=%.1fs)",
        sum(api) / total_seconds if total_seconds > 0 else float("inf"),
        sum(api),
        total_seconds,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_input = Path(__file__).with_name("candidates.json")
    p.add_argument("--input", type=Path, default=default_input)
    p.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("predictions_jev.json")
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--limit", type=int, default=None, help="Process only the first N captions."
    )
    p.add_argument(
        "--max-nouls-per-call",
        type=int,
        default=MAX_NOULS_PER_CALL,
        help="Split a candidate list into chunks of this many Noul questions per API call.",
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
            concurrency=args.concurrency,
            limit=args.limit,
            max_nouls_per_call=args.max_nouls_per_call,
        )
    )


if __name__ == "__main__":
    main()
