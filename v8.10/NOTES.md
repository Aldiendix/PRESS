# v8.10 (round 79, chained table)

v5/v6 code (19-feature singleton ranker + language split + link-post rule) with a **chained** table: v7's table
(rounds 40–77, 24,576 × 96, 4.68%) continued on rounds 40–78 (lr 0.005, 3,000 steps), packed at 19 bits per
character. 48,421 characters; tools/check.py PASS.

Evidence for chaining (tables never saw the test round): round 76 +0.0027 (continued from v4's table, vs a
from-scratch table on the same rounds); round 78 0.4337 for v6's table continued on rounds 40–77, the best of
all candidates and above the official round-78 top (0.4315), vs 0.4283 for v7.
