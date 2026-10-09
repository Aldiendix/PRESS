# v7.7

Variant of v7 for the round's portfolio: 17-feature ranker (no lexical ranker features; faster).

- Table: `cache/W_v7` (same training recipe as v7).
- Settings: social 120 clusters / 20% singletons; arXiv 80 clusters / 30% singletons; 17-feature ranker.
- Bench evidence vs v7's settings (unseen rounds): about -0.001 vs 19 features end to end (rounds 69-74), round 75 -0.0013.
- v7 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
