# v8.5

Variant of v8 for the round's portfolio: far fewer singletons (social 15%, arXiv 25%).

- Table: `cache/W_v8` (same training recipe as v8).
- Settings: social 120 clusters / 15% singletons; arXiv 80 clusters / 25% singletons; 19-feature ranker.
- Bench evidence vs v8's settings (unseen rounds): social: D -0.0019..-0.0044, A -0.0010, B+C +0.0003, round 76 -0.0033.
- v8 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
