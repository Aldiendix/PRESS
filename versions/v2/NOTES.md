# v2 (2026-10-06)

`solution.py` is 48,762 characters. Peak memory 531 MB for 5,000 texts (972 MB for 50,000); ~5 s per subset on one core.

**Changes from v1**
- **Self-training pass.** After the first clustering, a ridge classifier on the batch's exact TF-IDF vocabulary
  learns the first-pass clusters (90 clusters, α = 10); its scores are appended to the student embedding
  (weight 0.75) and the batch is clustered again. Adds ~6 s per subset.
- Table retrained at 3.3% non-zero (was 3.5%) to make room for the extra code. Same shape (24,576 × 96) and
  training data (rounds 40–73).

**Scores**
Honest estimate = identical configuration with the table trained on rounds 40–68, tested on rounds 69–73:

| Round | 69 | 70 | 71 | 72 | 73 | mean |
|---|---|---|---|---|---|---|
| v2 configuration, unseen rounds | 0.490 | 0.494 | 0.518 | 0.398 | 0.514 | 0.4825 |
| v1 configuration, unseen rounds | 0.478 | 0.481 | 0.507 | 0.395 | 0.510 | 0.474 |
| Official top score of that round | — | 0.469 | 0.473 | 0.393 | 0.495 | — |

The shipped table itself scores 0.4979 on rounds 69–73 (inside its training data).
Not yet submitted to Apex; no official score.
