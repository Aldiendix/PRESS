# v12.6

Dense table continued from W_v11_dchain on rounds 27-82 (lr 0.02, 6,000 steps).

- Table: `cache/W_v12_dchain` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 48733 characters; tools/check.py: RESULT: PASS
