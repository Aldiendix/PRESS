# v4 (2026-10-06, built during round 75)

`solution.py` is 49,113 characters; ~4 s and 531 MB per 5,000-text subset.

**Changes from v3**
- Cluster settings re-tuned with self-training switched on: social singleton share 0.30 → 0.20; arXiv titles
  80 → 180 clusters and singleton share 0.40 → 0.35.
- Table trained with sparsity annealed from 30% to 3.3% non-zero (rounds 40–74, as v3).
- Optional re-attachment of singletons to a near-identical centroid is implemented but switched off (no extra gain).

**Scores** — next-round protocol: table trained only on rounds before the test round.

| Test set (table trained through) | v3 settings | v4 settings |
|---|---|---|
| Rounds 69–73 (68) | 0.4825 | 0.4870 |
| Round 73 (72) | 0.5165 | 0.5233 |
| Round 74 (73) | 0.4123 | 0.4171 |

- Official top of round 74 was 0.3996.
- The shipped table has seen rounds 40–74, so it has no unseen-round score yet.
- Not yet submitted to Apex; no official score.

**Rebuilding with a newer round:** `tools/make_version.sh <N> <last finished round>` fetches the round, retrains the
table on rounds 40..last with this recipe and writes `versions/vN/solution.py`.
