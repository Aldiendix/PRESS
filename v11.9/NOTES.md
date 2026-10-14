# v11.9

Dense table, from scratch, seed 2, rounds 27-81 (all rounds in the current data style).

- Table: `cache/W_v11_d27` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 48142 characters; tools/check.py: RESULT: PASS
