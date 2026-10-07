# v4.2 (2026-10-07, built during round 75)

`solution.py` is 46,715 characters (3,285 spare); ~7 s and ~575 MB per 5,000-text subset on one core
(974 MB for 50,000 texts). Same table as v4 / v4.1 (rounds 40–74). Built with
`tools/stack/build.py --table cache/W_final4 --mini --out versions/v4.2/solution.py`.

**Changes from v4.1** (found and verified by the multi-agent campaign; details in `docs/RESEARCH_LOG.md`)
- **Learned singleton ranker.** Which points become singletons is decided by a 17-feature logistic score
  (density, self-training margins, cluster size, etc.) fitted offline on past rounds to the points whose removal
  raises the score, instead of by density alone. Same number of singletons.
- **Language split.** Clusters that mix scripts / languages are split by language.
- **Link-post rule.** Short captions of image links (`preview.redd.it`) that the pipeline scatters are grouped into
  one cluster when there are at least 20 of them.
- **Compact container.** Table stored with ANS entropy coding and the code minified (`tools/stack/mini_template.py`;
  readable equivalent in `src/stack/solution_template.py`). Decoded table is bit-identical.

**Scores** (every table and ranker weight held out from the test rounds; deltas vs v4.1)
| Check | v4.1 | v4.2 |
|---|---|---|
| Rounds 61–74 (56 subsets, table 40–60), end to end | 0.4333 | 0.4407 (+0.0074; odd +0.0070, even +0.0078; 53 better, 3 worse) |
| Rounds 69–73 (table 40–68) | 0.4874 | 0.4909 (+0.0036; 19 of 20 subsets better) |
| Round 73 (table 40–72) | 0.5248 | 0.5274 (+0.0027) |
| Round 74 (table 40–73) | 0.4179 | 0.4205 (+0.0026) |

Survived an adversarial verifier (independent re-implementation, perturbation, leakage checks). Expected gain in
deployment: about +0.0025 to +0.0035 per round (about +0.6%). Not yet submitted to Apex.
