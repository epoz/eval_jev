"""
Purpose: Compare Jev's probability-sorted Iconclass predictions against
         Claude's ranked predictions (§4.1/§4.3 in Plan.md) via rank
         correlation and top-K set overlap - sidesteps comparing the raw
         probability values, which aren't calibrated the same way between
         the two systems.
Usage:   python compare_rankings.py
         (reads the two prediction files below; no API calls, no cost)
Inputs:  data/output/predictions_jev.json   (Jev, probability-sorted per item)
         data/output/rankings_claude.json   (Claude, rank-ordered per item)
Outputs: data/output/jev_vs_claude_ranking_comparison.csv (per-item metrics)
         console summary (mean/median across items) + any code-set mismatches
Depends: scipy (spearmanr, kendalltau)
Assumes: both files are joined on (s, caption); both were independently
         verified (Plan.md §3, §4.3.3) to have full candidate coverage per
         item, so code-set mismatches are expected to be rare/absent, but
         are checked and reported explicitly rather than assumed away.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

PROJECT_DIR = Path(__file__).parent.parent
JEV_PATH = PROJECT_DIR / "data" / "output" / "predictions_jev.json"
CLAUDE_PATH = PROJECT_DIR / "data" / "output" / "rankings_claude.json"
OUTPUT_CSV = PROJECT_DIR / "data" / "output" / "jev_vs_claude_ranking_comparison.csv"

TOP_KS = (3, 5, 10)


def _key(s: int, caption: str) -> tuple[int, str]:
    return (s, caption)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def load_jev(path: Path) -> dict[tuple[int, str], dict]:
    """Score = raw probability (higher = more relevant), codes pre-sorted
    by probability descending for top-K."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for item in raw:
        codes = [p["code"] for p in item["predictions"]]
        scores = [p["probability"] for p in item["predictions"]]
        out[_key(item["s"], item["caption"])] = {"codes": codes, "scores": scores}
    return out


def load_claude(path: Path) -> dict[tuple[int, str], dict]:
    """Score = -rank, so higher = more relevant (same direction as Jev's
    probability) - rank 1 (best) becomes the largest score."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for item in raw:
        ranked = sorted(item["ranking"], key=lambda r: r["rank"])
        codes = [r["code"] for r in ranked]
        scores = [-float(r["rank"]) for r in ranked]
        out[_key(item["s"], item["caption"])] = {"codes": codes, "scores": scores}
    return out


def compare_item(jev: dict, claude: dict) -> dict:
    jev_score_by_code = dict(zip(jev["codes"], jev["scores"]))
    claude_score_by_code = dict(zip(claude["codes"], claude["scores"]))

    shared = sorted(set(jev_score_by_code) & set(claude_score_by_code))
    excluded = (set(jev_score_by_code) ^ set(claude_score_by_code))

    jev_shared = [jev_score_by_code[c] for c in shared]
    claude_shared = [claude_score_by_code[c] for c in shared]

    if len(shared) < 2:
        spearman = kendall_tau = float("nan")
    else:
        spearman = spearmanr(jev_shared, claude_shared).correlation
        kendall_tau = kendalltau(jev_shared, claude_shared).correlation

    row = {
        "n_shared_codes": len(shared),
        "n_excluded_codes": len(excluded),
        "spearman": spearman,
        "kendall_tau": kendall_tau,
    }
    for k in TOP_KS:
        jev_topk = {c for c in jev["codes"][:k]}
        claude_topk = {c for c in claude["codes"][:k]}
        row[f"jaccard_top{k}"] = _jaccard(jev_topk, claude_topk)
    row["excluded_codes"] = sorted(excluded)
    return row


def main() -> None:
    jev = load_jev(JEV_PATH)
    claude = load_claude(CLAUDE_PATH)

    jev_only = jev.keys() - claude.keys()
    claude_only = claude.keys() - jev.keys()
    if jev_only:
        print(f"WARNING: {len(jev_only)} items only in Jev's file (no Claude match)")
    if claude_only:
        print(f"WARNING: {len(claude_only)} items only in Claude's file (no Jev match)")

    shared_keys = sorted(jev.keys() & claude.keys(), key=lambda k: k[0])
    rows = []
    mismatched_items = []
    for s, caption in shared_keys:
        row = compare_item(jev[(s, caption)], claude[(s, caption)])
        row["s"] = s
        row["caption_snippet"] = caption[:60]
        if row["n_excluded_codes"]:
            mismatched_items.append((s, row["excluded_codes"]))
        rows.append(row)

    fieldnames = [
        "s", "caption_snippet", "n_shared_codes", "n_excluded_codes",
        "spearman", "kendall_tau", *[f"jaccard_top{k}" for k in TOP_KS],
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    def _summary(field: str) -> str:
        values = [r[field] for r in rows if not math.isnan(r[field])]
        if not values:
            return "n/a (no valid values)"
        return (
            f"mean={statistics.mean(values):.3f}  median={statistics.median(values):.3f}  "
            f"min={min(values):.3f}  max={max(values):.3f}  (n={len(values)}/{len(rows)})"
        )

    print("\n── summary across", len(rows), "items ──")
    print("spearman   :", _summary("spearman"))
    print("kendall_tau:", _summary("kendall_tau"))
    for k in TOP_KS:
        print(f"jaccard_top{k}: {_summary(f'jaccard_top{k}')}")

    if mismatched_items:
        print(f"\n{len(mismatched_items)} items had code-set mismatches (excluded from that item's correlation):")
        for s, excluded_codes in mismatched_items:
            print(f"  s={s}: {excluded_codes}")
    else:
        print("\nNo code-set mismatches - every item compared over its full shared candidate set.")


if __name__ == "__main__":
    main()
