# Evaluate Jev

An evaluation of [TypeSafe](https://typesafe.ai)'s **Jev** model against
general-purpose chat LLMs (Anthropic Claude) on one FIZ task: suggesting
**Iconclass** notations for cultural heritage images, based on the images'
text captions.

Jev is a "System One" model. It doesn't generate free text. Instead it answers
typed questions (Noul = yes/no, Choice, Score) with probabilities. The question
we're testing is whether this is faster, cheaper, and good enough compared to
asking a chat LLM and parsing its output.

> Status: work in progress. The results so far measure **speed, cost and
> agreement between systems**, not correctness. There is no ground truth
> dataset yet (see [Open work](#open-work)).

## Task

Each item has an image caption and a list of candidate Iconclass codes
(4–61 per caption, about 23 on average). The candidates were found by
embedding-similarity search over Iconclass. Each system has to judge or rank
which candidates apply to the caption. More than one concept can apply to
the same caption, so this is a multi-label problem.

```json
{"s": 9816610358951523578,
 "caption": "A 16th-century woodcut historiated initial letter 'H'. ...",
 "IC": [["11E1(+3)", "Holy Ghost represented as a dove ..."], ["49L171", "historiated initial"], ...]}
```

Dataset: 92 captions, 2,126 caption–candidate pairs (`data/candidates.json`).
The results below were computed on an earlier version with 97 captions and
2,265 pairs (`data/candidates.with-duplicates.json`), in which 5 images appear
twice with different captions.

## Systems and approach

| System | How it's asked |
|---|---|
| **Jev** (`jev-1.13.0`) | One Noul per candidate ("does this caption match this Iconclass concept?"), sent together in one `system_one` call, max 30 per call. The answer is P(yes) for each candidate. |
| **Claude Haiku 4.5 / Opus 5**, classification | Same chunking (30 candidates per call). Claude returns a probability for each code in structured output. |
| **Claude Haiku 4.5**, ranking | One call per caption. Claude returns the candidate codes ranked from most to least likely, with no probabilities. |

Claude's "probabilities" aren't calibrated the way Jev's are. So the systems
are compared on how they **rank** the candidates, not on the raw numbers.

## Results so far

### Speed and cost (97 captions, 123 calls, concurrency 4)

| | Jev | Claude Haiku 4.5 | Claude Opus 5 |
|---|---|---|---|
| total wall time | 9.96 s | 74.83 s | 235.31 s |
| mean time per call | 0.315 s | 2.396 s | 7.587 s |
| cost of the run | not computed (only input tokens are billed, $0.042 per million tokens) | ~$0.37 | ~$2.92 |

Jev was about 7.5× faster than Haiku and about 24× faster than Opus. The
ranking version of the Claude Haiku run cost ~$0.17.

### Agreement between Jev and Claude's ranking (n = 97)

| Metric | mean | median |
|---|---|---|
| Spearman ρ | 0.558 | 0.609 |
| Kendall τ | 0.420 | 0.441 |
| Jaccard, top 3 | 0.366 | 0.500 |
| Jaccard, top 10 | 0.634 | 0.538 |

The two systems agree moderately. They agree more on which part of the
candidate list is relevant than on the single best code. We saw two ways
they disagree:

- **Near-synonym candidates.** Several codes describe variations of the same
  scene, and the two systems pick different ones.
- **Generic vs. specific codes.** Jev tends to prefer broad top-level codes,
  while Claude prefers the most specific code even when the caption only
  weakly supports it.

## Project layout

| Path | Contents |
|---|---|
| `data/candidates.json` | The evaluation set, deduplicated: 92 captions, 2,126 caption–candidate pairs |
| `data/candidates.with-duplicates.json` | The earlier 97-item set (5 images captioned twice). The outputs in `data/output/` and the results above were computed on this one |
| `data/ids.tsv` | Caption ids |
| `data/output/` | Predictions, rankings and comparison CSV for each system |
| `scripts/gen_candidates.py` | Builds candidate lists (EmbeddingGemma + DuckDB vector search) |
| `scripts/classify_captions_jev.py` | Jev classification using Noul questions, with timing report |
| `scripts/classify_captions_claude.py` | Claude classification, chunked like the Jev script, same timing format |
| `scripts/rank_captions_claude.py` | Claude ranking version (codes only, cheapest) |
| `scripts/compare_rankings.py` | Spearman, Kendall and top-K Jaccard between Jev and Claude |
| `scripts/score_iconclass.py` | Early experiment using Jev's Score question type |

## Running

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
export TYPESAFE_API_KEY=...      # Jev
export ANTHROPIC_API_KEY=...     # Claude

uv run scripts/classify_captions_jev.py --input data/candidates.json
uv run scripts/classify_captions_claude.py --limit 3        # check a few first; this costs money
uv run scripts/rank_captions_claude.py --limit 3
uv run scripts/compare_rankings.py                         # runs locally, no API calls
```

## Open work

- **Ground truth.** A labelled set is needed before we can measure accuracy
  instead of just agreement.
- A cost-vs-quality plot (Jev / Haiku / Opus), once there is ground truth.
- Bootstrap confidence intervals for the agreement numbers.
- Add a yes/no (Noul) task and a single-label (Choice) task.
- Combine the scripts into a single pipeline.
