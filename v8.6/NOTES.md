# v8.6

Variant of v8 for the round's portfolio: finer social cut (160 clusters).

- Table: `cache/W_v8` (same training recipe as v8).
- Settings: social 160 clusters / 20% singletons; arXiv 80 clusters / 30% singletons; 19-feature ranker.
- Bench evidence vs v8's settings (unseen rounds): social: D -0.0017, A -0.0008, E +0.0018.
- v8 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
