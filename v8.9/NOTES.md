# v8.9

Variant of v8 for the round's portfolio: table seed 2 + fewer singletons.

- Table: `cache/W_v8_s2` (same training recipe as v8).
- Settings: social 120 clusters / 17% singletons; arXiv 80 clusters / 25% singletons; 19-feature ranker.
- Bench evidence vs v8's settings (unseen rounds): combination of v.2 and v.4.
- v8 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
