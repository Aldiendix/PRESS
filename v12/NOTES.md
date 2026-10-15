# v12

Wider table (40,960 rows), from scratch, seed 0, rounds 27-82.

- Table: `cache/W_v12` (40,960 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 49557 characters; tools/check.py: RESULT: PASS
