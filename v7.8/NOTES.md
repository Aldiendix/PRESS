# v7.8

Variant of v7 for the round's portfolio: table seed 1 + more singletons.

- Table: `cache/W_v7_s1` (same training recipe as v7).
- Settings: social 120 clusters / 23% singletons; arXiv 80 clusters / 35% singletons; 19-feature ranker.
- Bench evidence vs v7's settings (unseen rounds): combination of v.1 and v.3.
- v7 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
