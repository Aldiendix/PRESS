# v7.9

Variant of v7 for the round's portfolio: table seed 2 + fewer singletons.

- Table: `cache/W_v7_s2` (same training recipe as v7).
- Settings: social 120 clusters / 17% singletons; arXiv 80 clusters / 25% singletons; 19-feature ranker.
- Bench evidence vs v7's settings (unseen rounds): combination of v.2 and v.4.
- v7 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
