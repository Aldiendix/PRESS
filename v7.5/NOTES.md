# v7.5

Variant of v7 for the round's portfolio: far fewer singletons (social 15%, arXiv 25%).

- Table: `cache/W_v7` (same training recipe as v7).
- Settings: social 120 clusters / 15% singletons; arXiv 80 clusters / 25% singletons; 19-feature ranker.
- Bench evidence vs v7's settings (unseen rounds): social: D -0.0019..-0.0044, A -0.0010, B+C +0.0003, round 76 -0.0033.
- v7 itself has the best expected score; variants trade a little expected score for different behaviour on rounds with unusually low or high reference noise.
- tools/check.py: RESULT: PASS
