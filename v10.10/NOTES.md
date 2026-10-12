# v10.10

Dense table (24,576 rows, 4.68%), from scratch, seed 0, trained on ALL rounds 1-80 (more training data).

- Table: `cache/W_v10_all` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 20% singletons, arXiv 80 clusters / 30%.
- 48047 characters; tools/check.py: RESULT: PASS
