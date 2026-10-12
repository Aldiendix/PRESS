# v10.5

Wider table continued from W_v9 on rounds 40-80 (lr 0.005, 3,000 steps).

- Table: `cache/W_v10_wchain` (40,960 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%.
- 49359 characters; tools/check.py: RESULT: PASS
