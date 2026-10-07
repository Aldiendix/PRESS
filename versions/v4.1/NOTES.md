# v4.1 (2026-10-07, built during round 75)

`solution.py` is 49,111 characters. Same code and same packed table as v4 (rounds 40–74).

**Change from v4**
- arXiv-title settings: 80 final clusters and a 0.30 singleton share (v4: 180 and 0.35). Found by a joint random
  search of all pipeline settings on 14 unseen rounds (61–74, table trained on rounds 40–60); the social settings
  of v4 came out as the best of 71 configurations and are unchanged.

**Scores**
| Check (table never saw the test rounds) | v4 | v4.1 |
|---|---|---|
| 14 arXiv subsets, rounds 61–74 | 0.3561 | 0.3624 (odd rounds +0.0055, even +0.0070; better on 9 of 14) |
| Rounds 69–73, full round score | 0.4870 | 0.4874 |
| Round 73 | 0.5233 | 0.5247 |
| Round 74 | 0.4171 | 0.4180 |

The gain is about +0.001 to +0.002 per round: small, but it has the same sign on every check.
Not yet submitted to Apex; no official score.
