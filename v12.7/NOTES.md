# v12.7

Leader-lineage table continued from W_v11_chainL on rounds 27-82 (lr 0.005, 3,000 steps), 15% singletons.

- Table: `cache/W_v12_chainL` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 15% singletons, arXiv 180 clusters / 25%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 49363 characters; tools/check.py: RESULT: PASS
