# v4.3 (2026-10-07, built during round 75)

`solution.py` is 47,009 characters; ~12 s and ~640 MB per 5,000-text social subset on one core (5 s for arXiv;
974 MB for 50,000 texts). Same table as v4–v4.2 (rounds 40–74). Built with
`tools/stack/build.py --table cache/W_final4 --mini --template tools/stack/mini19_template.py`.

**Change from v4.2:** the singleton ranker has two more features, both from the batch's exact TF-IDF space:
the share of a text's 10 most similar texts that sit in its cluster, and the cosine to its own cluster's TF-IDF
centroid minus the best other centroid. Weights fitted offline on rounds 61–74.

**Scores** (held-out tables and weights; deltas vs v4.1)
| Check | v4.1 | v4.2 | v4.3 |
|---|---|---|---|
| Rounds 61–74, bench (56 subsets) | — | +0.0076 | +0.0076 + ~0.002 on the ranker part |
| Rounds 69–73 (table 40–68), end to end | 0.4874 | 0.4912 | 0.4921 |
| Round 73 (table 40–72) | 0.5248 | 0.5276 | 0.5286 |
| Round 74 (table 40–73) | 0.4179 | 0.4205 | 0.4213 |

Both the stack and the 19-feature ranker survived adversarial verification (independent re-implementations,
weights perturbed ±25%, leakage checks). The compact build gives partitions identical to the verifier-checked
readable build. Expected gain over v4.1: about +0.003 to +0.004 per round. Not yet submitted to Apex.
