# v9.1

Variant of v9 for the round's portfolio: table seed 1, default settings.

- Table: `cache/W_v9_s1` (same training recipe as v9).
- Settings: social 120 clusters / 20% singletons; arXiv 80 clusters / 30% singletons; 19-feature ranker.
- Bench evidence vs v9's settings (unseen rounds): seeds are equal on average (D: 0.4419 / 0.4425 / 0.4441); differs per round.
- v9 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
