# v10.6

Dense table continued from W_v8_chain on rounds 40-80 (lr 0.02, 6,000 steps).

- Table: `cache/W_v10_dchain` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%.
- 48590 characters; tools/check.py: RESULT: PASS
