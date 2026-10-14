# v11

Wider table (40,960 rows), from scratch, seed 0, rounds 40-81.

- Table: `cache/W_v11` (40,960 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 49578 characters; tools/check.py: RESULT: PASS
