# v11.4

Dense table, from scratch, seed 1, rounds 40-81.

- Table: `cache/W_v11_d1` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 48239 characters; tools/check.py: RESULT: PASS
