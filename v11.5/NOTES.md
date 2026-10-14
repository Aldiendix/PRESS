# v11.5

Wider table continued from W_v10 on rounds 40-81 (lr 0.005, 3,000 steps).

- Table: `cache/W_v11_wchain` (40,960 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 49497 characters; tools/check.py: RESULT: PASS
