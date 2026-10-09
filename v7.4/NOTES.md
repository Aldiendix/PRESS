# v7.4

Variant of v7 for the round's portfolio: fewer singletons (social 17%, arXiv 25%), like the round-76 leader.

- Table: `cache/W_v7` (same training recipe as v7).
- Settings: social 120 clusters / 17% singletons; arXiv 80 clusters / 25% singletons; 19-feature ranker.
- Bench evidence vs v7's settings (unseen rounds): social: D -0.0016..-0.0037, B+C +0.0005, E -0.0017, round 76 -0.0031; arXiv: D -0.0011..-0.0018, B+C +0.0044, E +0.0014.
- v7 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
