# v8.3

Variant of v8 for the round's portfolio: more singletons (social 23%, arXiv 35%).

- Table: `cache/W_v8` (same training recipe as v8).
- Settings: social 120 clusters / 23% singletons; arXiv 80 clusters / 35% singletons; 19-feature ranker.
- Bench evidence vs v8's settings (unseen rounds): social: D +0.0011..+0.0016, A +0.0001, E -0.0049, round 76 +0.0005; arXiv: D -0.0001..+0.0013, round 76 -0.0024.
- v8 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
