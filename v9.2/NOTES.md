# v9.2

Variant of v9 for the round's portfolio: table seed 2, default settings.

- Table: `cache/W_v9_s2` (same training recipe as v9).
- Settings: social 120 clusters / 20% singletons; arXiv 80 clusters / 30% singletons; 19-feature ranker.
- Bench evidence vs v9's settings (unseen rounds): as v.1.
- v9 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
