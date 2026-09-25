"""
Purpose: For each captioned image in candidates.json, ask Jev to Score how well
         each candidate iconclass code corresponds to the caption.
Usage:   TYPESAFE_API_KEY=... ./.venv/bin/python3 score_iconclass.py [--limit N]
Inputs:  /Users/mta/Documents/claude/notes/fiz/fizfiles-unison/Evaluate Jev/candidates.json
         List of {"s": <id>, "caption": <str>, "IC": [[code, label], ...]}
Outputs: Printed per-item ranking of codes by Score (descending), test run only
         (no file written yet).
Depends: typesafe_sdk (see ./.venv), TYPESAFE_API_KEY env var
Assumes: candidate IC lists can be up to ~60 entries per item, so each item's
         codes are chunked into groups of 20 to keep requests a manageable size.
"""

import argparse
import json

from typesafe_sdk import Score, TypeSafeClient

CANDIDATES_PATH = (
    "/Users/mta/Documents/claude/notes/fiz/fizfiles-unison/Evaluate Jev/candidates.json"
)
CHUNK_SIZE = 20
# 3 ordered levels (indices 0, 1, 2) -> response.answers[code].score is the
# probability-weighted position across them, so it ranges 0-2, not 0-1.
# e.g. ~1.99 means near-certain "Directly depicted".
SCORE_CRITERIA = ["Not related", "Loosely related", "Directly depicted"]


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def score_item(client, caption, ic_codes):
    scores = {}
    for chunk in chunked(ic_codes, CHUNK_SIZE):
        questions = {
            code: Score(
                instructions=(
                    f"Does the iconclass code {code} (\"{label}\") "
                    "correspond to the caption?"
                ),
                criteria=SCORE_CRITERIA,
            )
            for code, label in chunk
        }
        response = client.system_one(state=caption, questions=questions)
        for code, _ in chunk:
            scores[code] = response.answers[code].score
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)

    client = TypeSafeClient()

    for item in candidates[: args.limit]:
        scores = score_item(client, item["caption"], item["IC"])
        labels_by_code = dict(item["IC"])
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        print(f"\n=== s={item['s']} ===")
        print(item["caption"][:200] + ("..." if len(item["caption"]) > 200 else ""))
        for code, score in ranked:
            print(f"  {score:.2f}  {code}  {labels_by_code[code]}")


if __name__ == "__main__":
    main()
