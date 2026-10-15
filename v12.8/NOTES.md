# v12.8

Same table as v12.7 (leader-lineage chain), social 25% singletons (+0.0015 on 97 unseen subsets vs 20%).

- Table: `cache/W_v12_chainL` (24,576 rows x 96), 19-bit packing.
- Pipeline: v5/v6 stack (19-feature singleton ranker + language split + link-post rule); social 25% singletons, arXiv 80 clusters / 30%; early-style social batches (10th-percentile length < 80, median >= 200) use 24 clusters / 5% singletons.
- 49361 characters; tools/check.py: RESULT: PASS
