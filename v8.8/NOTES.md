# v8.8

Variant of v8 for the round's portfolio: table seed 1 + more singletons.

- Table: `cache/W_v8_s1` (same training recipe as v8).
- Settings: social 120 clusters / 23% singletons; arXiv 80 clusters / 35% singletons; 19-feature ranker.
- Bench evidence vs v8's settings (unseen rounds): combination of v.1 and v.3.
- v8 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
