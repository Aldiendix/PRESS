# v3 (2026-10-06, round 75)

`solution.py` is 48,749 characters. Same code as v2; only the packed table changed.

**Changes from v2**
- Table retrained on rounds 40–74 (v2: 40–73), so it includes the newest round.
- The 34 real arXiv subsets with cached teacher embeddings also contribute a teacher-similarity loss (weight 1);
  on unseen rounds this was neutral (0.4846 vs 0.4830).

**Scores**
- No unseen round exists yet for this table. Its v2 twin scored **0.4126 on round 74** while that round was
  unseen (official top of round 74: 0.3996; round-73 leader: 0.3954).
- Sanity check on rounds inside its training data: round 73 0.524, round 74 0.416.
- Not yet submitted to Apex; no official score.
