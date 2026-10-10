# v7.10 (round 78 addition)

v5/v6 code (19-feature singleton ranker + language split + link-post rule) with a **wider table**: 40,960 rows
(8,192 word + 32,768 character n-gram buckets) × 96, 2.48% non-zero, trained on rounds 40–77, packed at 19 bits
per character. 49,450 characters; tools/check.py PASS.

Evidence (tables never saw the test rounds):
- Set D (rounds 61–74, tables trained on 40–60, two seeds): 0.4476 vs 0.4444 for v7's shape (+0.003).
- Round 77 (table trained on 40–76), end to end: **0.5390** vs v6.9 0.5359 (our round-77 winner), v7's recipe
  0.5374, v6 0.5299; leader's round-77 table in our pipeline 0.5370 (bench).
