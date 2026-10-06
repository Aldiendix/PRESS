# v1 (2026-10-06)

First PRESS version. `solution.py` is 49,486 characters.

**Method**
- Hashed word 1–2-grams (8,192 buckets) + character 3–5-grams (16,384 buckets), `log1p`, batch IDF.
- Packed ternary student table, 24,576 × 96, 3.5% non-zero, per-column and per-row scales (80 KB as text).
  Trained with quantization in the loop on rounds 40–73: supervised contrastive loss on ground-truth cluster
  labels plus a neighbour-matching loss on the published x/y/z coordinates.
- k-NN smoothing (k = 20 social / 35 titles, α = 0.4, 4 iterations).
- Average-linkage clustering on cosine distance (120 clusters social / 80 titles).
- 30% (social) / 40% (titles) lowest-density points returned as singleton clusters.
- Runs in ~3 s per 5,000-text subset on one core.

**Scores**
The shipped table has seen rounds 40–73, so its honest estimate comes from the identical configuration trained
on rounds 40–68 and tested on rounds 69–73:

| Round | 69 | 70 | 71 | 72 | 73 | mean |
|---|---|---|---|---|---|---|
| v1 configuration, unseen rounds (3 seeds: 0.4743 / 0.4741 / 0.4712) | 0.478 | 0.481 | 0.507 | 0.395 | 0.510 | 0.474 |
| Best revealed reference (`5C55Guoe` v1) | 0.441 | 0.411 | 0.441 | 0.353 | 0.475 | 0.424 |
| Official top score of that round | — | 0.469 | 0.473 | 0.393 | 0.495 | — |

The shipped table itself scores 0.4985 on rounds 69–73 (inside its training data).
Not yet submitted to Apex; no official score.
