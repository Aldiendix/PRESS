# v9.9

Variant of v9 for the round's portfolio: table seed 2 + fewer singletons.

- Table: `cache/W_v9_s2` (same training recipe as v9).
- Settings: social 120 clusters / 17% singletons; arXiv 80 clusters / 25% singletons; 19-feature ranker.
- Bench evidence vs v9's settings (unseen rounds): combination of v.2 and v.4.
- v9 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
