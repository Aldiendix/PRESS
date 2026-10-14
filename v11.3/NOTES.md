# v11.3

Dense table (24,576 rows, 4.68%), from scratch, seed 0, rounds 40-81.

- Table: `cache/W_v11_d0` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 48192 characters; tools/check.py: RESULT: PASS
