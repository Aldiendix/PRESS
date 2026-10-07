# v5 (2026-10-07, for round 76)

`solution.py` is 46,985 characters; ~7–12 s and ~640 MB per 5,000-text subset on one core (973 MB for 50,000
texts). Built with `tools/make_version.sh 5 75 19`: the v4.3 code (19-feature singleton ranker + language split +
link-post rule, compact container) with the table retrained on rounds 40–75.

**Why it should beat v4.3:** each newer round in training has been worth about +0.005 on the next round.

**Unseen-round evidence for the code (round 75, table trained through round 74, never saw round 75):**
| File | Round 75 | vs official top of round 75 (0.4518) |
|---|---|---|
| v4.1 | 0.4513 | −0.0005 |
| v4.2 (17-feature stack) | 0.4523 | +0.0005 |
| v4.3 (19-feature stack) | 0.4536 | +0.0018 |

v5 itself has no unseen-round score yet (it has seen round 75: 0.4652 there, 0.4327 on round 74, both inside
training). Checks: NFC-stable, decoded table bit-identical, HTTP server health / 5,000-text / tiny / empty requests OK.
Not yet submitted to Apex.
