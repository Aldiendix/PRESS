# Research log

All scores are the official metric, `(max(0, ARI) + NMI) / 2` averaged over a round's four subsets, computed
locally with `tools/evaluate.py` against the labels in the public round files. "Unseen" means the model's
table was trained only on earlier rounds.

## 2026-10-05 / 06 — baseline study and v1

### Facts about the task
- Round files (`text, label, x, y, z`) include the ground-truth labels; 264 labelled subsets exist for rounds 1–73.
- The smallest ground-truth cluster always has ≥ 25 points (HDBSCAN `min_cluster_size=25`). Recent rounds have
  23–49 clusters and 10–30% noise per subset; rounds before ~30 had far less noise.
- Re-running the real teacher pipeline (all-mpnet-base-v2 → UMAP 15 nn / 5 dims → HDBSCAN) with a different
  UMAP seed scores ~0.75 on round 73. That is the practical ceiling.
- `x, y, z` are a separate 3-D UMAP of the teacher embeddings (HDBSCAN on them does not reproduce the labels),
  but they still carry neighbourhood information usable as a training signal.

### What limits the public solutions
- On the best revealed solution's features, real UMAP + HDBSCAN scores 0.43; its own agglomerative clustering
  scores 0.475. On real teacher embeddings the order flips (0.76 vs 0.41–0.45). The embedding is the bottleneck.
- Ground-truth noise is predicted by neighbour-label disagreement (AUC ~0.93 in teacher space), not by density
  (AUC 0.57–0.78). With weak embeddings neither signal works (AUC ~0.56–0.68).

### Student table experiments (unseen rounds 71–73, table alone, quick clusterer)
`purity` = share of a point's 15 nearest neighbours with the same ground-truth cluster.

| Table | Size | social purity | arXiv purity |
|---|---|---|---|
| Reference social / title tables | 23 KB / 36 KB | 0.45 | 0.49 |
| Float, 12,288 × 48, label-supervised contrastive loss | — | 0.61 | 0.53 |
| Float, 12,288 × 96 | — | 0.645 | 0.55 |
| Float, 49,152 × 48 | — | 0.658 | 0.56 |
| Float, word n-grams only (no char n-grams) | — | 0.555 | 0.46 |
| Ternary QAT, 12,288 × 48, 12% non-zero | 48 KB | 0.558 | 0.45 |
| + batch IDF, per-row scale, x/y/z neighbour loss | 42 KB | 0.570 | 0.46 |
| Ternary, 16,384 × 64, 11% non-zero | 78 KB | 0.625 | 0.51 |
| Ternary, 24,576 × 96, 3.5% non-zero | 79 KB | 0.645 | 0.535 |
| Same, trained on rounds 30–68 only | 76 KB | 0.657 | 0.548 |

- More hash buckets and dimensions with sparser weights is the efficient direction at a fixed size.
- Product quantization after training (0.52–0.54 at 60 KB) is worse than ternary quantization-aware training.
- Training three times longer, temperature 0.07 / 0.15, and dropping noise points as negatives: no gain.
- Recent rounds only (30+ or 40+) beats training on all rounds.

### Pipeline experiments
- k-NN smoothing of the embeddings is worth ~+0.1 (0.39 → 0.51 on social subsets with the float table).
- Adding batch-fitted views (TF-IDF LSA, retrofitted word vectors) to the student: +0.001 to +0.003. Not used.
- Removing top principal components: hurts (−0.02 to −0.06).
- Real UMAP + HDBSCAN on the student embedding: 0.45 vs 0.47 for smoothing + average linkage. Not used.
- Cluster count (50–200), singleton share (0.2–0.4), small-cluster handling: all within ±0.003.
- Self-training (ridge classifier on exact batch TF-IDF, trained on the initial clusters, scores appended as
  features): +0.005 to +0.012 on the first small table. Candidate for v2.
- Keeping URL / mention markers and punctuation tokens: better on the quick proxy, not better end to end.

### Packing
- 15 bits per character (code points from U+4E00). LZMA on a 3%-dense bit mask wastes ~12%; Golomb-Rice coding
  of the gaps between non-zero weights is within ~1% of the entropy and needs no decompressor for the mask.

### End-to-end results on unseen rounds 69–73
| Solution | 69 | 70 | 71 | 72 | 73 | mean |
|---|---|---|---|---|---|---|
| Reference `5C55Guoe` v1 (round 73) | 0.441 | 0.411 | 0.441 | 0.353 | 0.475 | 0.424 |
| Reference `5Eh8oDoA` v1 (round 72; rounds ≤ 71 likely in its training data) | 0.464 | 0.469 | 0.472 | 0.393 | 0.434 | 0.447 |
| PRESS prototype, 12,288 × 48 table (48 KB) | 0.430 | 0.417 | 0.457 | 0.351 | 0.491 | 0.429 |
| PRESS, 16,384 × 64 table | 0.456 | 0.460 | 0.501 | 0.388 | 0.506 | 0.462 |
| PRESS, 24,576 × 96 table, trained on rounds 30–68 | 0.469 | 0.470 | 0.501 | 0.396 | 0.511 | 0.470 |
| PRESS, same shape, trained on rounds 40–68 | 0.478 | 0.481 | 0.507 | 0.395 | 0.510 | 0.474 |

Official top score of round 73 at the time: 0.4949 (code not yet revealed).

## 2026-10-06 — towards v2

### Self-training at test time (adopted)
After the first clustering pass, a ridge classifier is fitted on the batch's exact TF-IDF vocabulary (word 1–2-grams
+ char 3–5-grams, no hashing) to predict the first-pass clusters of the dense points; its centred, normalised
decision scores are appended to the student embedding and the pipeline is run again.

| Table (trained on rounds 40–68), unseen rounds 69–73 | 69 | 70 | 71 | 72 | 73 | mean |
|---|---|---|---|---|---|---|
| without self-training | 0.478 | 0.481 | 0.507 | 0.395 | 0.510 | 0.474 |
| self-training (60 clusters, α = 10, weight 0.5) | 0.487 | 0.490 | 0.516 | 0.399 | 0.513 | 0.481 |
| self-training (90 clusters, α = 10, weight 0.75) | 0.490 | 0.494 | 0.518 | 0.398 | 0.514 | 0.4825 |

- Seed-to-seed standard deviation of the mean is ~0.002, so the gain (+0.008) is real.
- Two or three self-training iterations, 40–90 first-pass clusters, α 5–10, weight 0.5–1.0: all within ±0.002.
- Cost: ~6 s per subset on one core.

### Things that did not help
| Change | Result (mean, unseen rounds 69–73) |
|---|---|
| Train through the k-NN smoothing step (smoothing inside the loss) | 0.4734 vs 0.474 |
| 128 dimensions instead of 96 at the same size | 0.4730 |
| Table trained on arXiv subsets only | arXiv 0.360 vs 0.354; not worth splitting the budget |
| Synthetic arXiv subsets (public 2025–26 titles labelled with all-mpnet-base-v2 + UMAP + HDBSCAN, min_samples 5–8) added to training | 0.4666 with teacher-similarity loss, 0.4647 without; arXiv fell to 0.346 |
| x/y/z loss weight 0.25 / 1.0, learning rate 0.04, 40,960 × 64 table (all with self-training) | 0.482 / 0.484 / 0.483 / 0.4845 — within noise of 0.4825 |

- Calibration by-product: on six real arXiv subsets, UMAP(15 nn, 5 dims, min_dist 0, cosine) + HDBSCAN(min size 25,
  **min_samples 8**, EOM) agrees best with the ground truth (score 0.644, 36 clusters vs 38, noise 30% vs 27%).
- The synthetic-data result suggests the student gains from the recurring structure of the competition's own
  rounds; outside data dilutes it even when it is from the same source.

### Where the approach saturates (all with self-training, unseen rounds 69–73)
| Experiment | Result |
|---|---|
| Table size 47 KB / 77 KB / 128 KB / 207 KB / unquantized float | 0.476 / 0.484 / 0.487 / 0.491 / 0.493 |
| Sparsity annealed from dense to 3.3% during training | 0.4834–0.4845 vs 0.4825 (noise level) |
| Residual MLP (512 hidden) on top of the float table | worse than linear (neighbour purity 0.62 vs 0.69): overfits |
| Teacher-embedding similarity loss on the 34 real arXiv subsets, weight 1 / 3 | 0.4846 / 0.4769 (arXiv 0.359 / 0.349 vs 0.357) |
| Smoothing variants: similarity-weighted, mutual-k-NN, re-computed graph, shared-neighbour ranking | 0.4825 / 0.4724 / 0.4559 / 0.4596 vs 0.4838 for the plain mean |
| Oracle per-subset cluster count and singleton share | +0.006 social, +0.002 arXiv at most |
| Shared `-1` cluster for the most noise-like 2–20% of points (7 scores tried) | always worse; precision of every score is ~0.3 |
| Supervised noise detector from text (rounds 40–68 → 69–73) | AUC 0.66; precision at top 2%: 0.34 social, 0.55 arXiv |

- The shipped table scores 0.498 even on rounds inside its training data, against 0.4825 on unseen rounds: the
  bag-of-n-grams model class, not the bit budget or the amount of data, is what limits the score.
- The platform's client measures the limit with Python `len()` (code points). Characters outside the BMP would
  carry 20 bits instead of 15 (+33% table), but the table-size curve says that is worth only ~+0.002, and it is
  untested on the server side. Not used.

### Robustness of the shipped file
- 5,000 texts: ~5 s on one core, 531 MB peak. 50,000 texts: ~12 s, 972 MB (limit 1.5 GB; batches above 6,000
  texts are clustered on a 6,000-text sample and the rest take the majority label of their 5 nearest sampled points).
- Empty or symbol-only texts get `-1`; batches of 1–2 texts return distinct ids; an empty request returns HTTP 400.

## 2026-10-06 06:50 UTC — round-73 leader code revealed (`5Eh8oDoA` v3, official 0.4949)

Local replay (`reference/r73_5Eh8oDoA_v3_0.4949.py`, 49,396 characters, ~20 s per subset):

| Solution | 69 | 70 | 71 | 72 | 73 | mean |
|---|---|---|---|---|---|---|
| Leader v3 (rounds before 73 are probably in its tables' training data) | 0.487 | 0.489 | 0.514 | 0.391 | 0.495 | 0.475 |
| PRESS v2 configuration, table trained on rounds 40–68 (all five rounds unseen) | 0.490 | 0.494 | 0.518 | 0.398 | 0.514 | 0.4825 |

Round 73 is the only round both have not seen: 0.514 vs 0.495 (+3.8%). Round 74 data is not public until the round ends.

**What changed versus `5C55Guoe` v1 (0.4749)** — same architecture and packed tables (the social table differs slightly):
- More hand-tuned regimes keyed on median text length and its 10th percentile (view weights, number of
  principal components removed, spectral dimensions 22–56, fixed cluster counts 28/32, noise share 0.10–0.25).
- `LM`: merge clusters whose words are < 12% English stop-words (non-English or spam-like) when their centroids
  have cosine ≥ 0.9.
- `MNN`: merge pairs of tiny clusters that are mutual nearest neighbours (cosine ≥ 0.8 / 0.86).
- Clusters holding more than 15% of the points are re-split with a finer cut of the same linkage tree.
- Title pipeline: smoothing strength 0.4 → 0.6, k 35 → 20.

**Tested on the PRESS pipeline (unseen rounds 69–73, baseline 0.4825)**
| Idea | Result |
|---|---|
| Re-split clusters above 15% / 10% of points (240 or 400 fine clusters) | 0.4825 / 0.4800 / 0.4820 |
| Re-attach singletons with cosine ≥ 0.95 / 0.9 / 0.8 to a cluster centroid | 0.4840 / 0.4839 / 0.4749 |

Nothing adopted: the gains are inside the seed noise (~0.002). The leader's improvements are regime tuning on
top of a weaker embedding; none of it addresses embedding quality, which is where PRESS is ahead.

## 2026-10-06 15:40 UTC — round 74 (unseen by every model)

Round 74 ended; its data was not available to any solution below when it was built.

| Solution | subset 1 | subset 2 | subset 3 | arXiv | round |
|---|---|---|---|---|---|
| **PRESS v2** | 0.373 | 0.376 | 0.512 | 0.390 | **0.4126** |
| PRESS v1 | 0.363 | 0.368 | 0.481 | 0.384 | 0.3990 |
| Official top score of round 74 (`5H5t7Dkm` v4) | | | | | 0.3996 |
| Round-73 leader `5Eh8oDoA` v3 (local replay = its official round-74 score) | 0.341 | 0.344 | 0.527 | 0.369 | 0.3954 |
| `5C55Guoe` v1 | 0.313 | 0.339 | 0.429 | 0.379 | 0.3648 |

- v2 is 3.3% above the round's official top (1% is required to take the lead) and matches the +0.008–0.014 gain of
  self-training over v1 seen on rounds 69–73.
- Subsets 1 and 2 had 26–27% noise and score low for everyone.
- Round 75 opened at 15:30 UTC with a top score of 0.4410 (score to beat 0.4454); PRESS has not been submitted.

## 2026-10-06 evening — after round 75 opened

Round 75 leaderboard at 17:00 UTC: top 0.4518, with a cluster of new entries at 0.4475–0.4518 submitted from
16:00 onward, shortly after v2/v3 became public in this repository.

### Error decomposition of the pipeline (unseen rounds 69–73, baseline 0.4838)
| Hypothetical | Score |
|---|---|
| My clusters, ground-truth noise known exactly (as one cluster) | 0.7535 |
| Ground-truth clusters, my singleton choice | 0.6777 |
| Ground-truth clusters, noise points labelled by my clusters | 0.7888 |
| No singletons at all | 0.4700 |
| Same number of singletons drawn at random with 100% noise precision | 0.4321 |
| Shared `-1` cluster of that size at precision 0.6 / 0.7 / 0.85 | 0.4717 / 0.5064 / 0.5268 |

- Most of the remaining loss is noise identification, but a shared noise cluster needs ≥ 0.7 precision at ~30%
  volume; every signal tried reaches ~0.3 (a cross-validated GBM on 12 signals: AUC 0.705), and even re-running the
  real teacher pipeline only agrees with the ground-truth noise at ~0.7 precision.
- Density-based singletons work as abstention on badly clustered points (73% of them are noise or misassigned),
  not as noise detection. A learned abstention ranker (logistic / GBM, AUC 0.79 for "noise or misassigned") scores
  0.480–0.481, below the plain density rule.

### Next-round protocol (train through round T, test on T+1)
| Training rounds | Round 73 | Round 74 |
|---|---|---|
| 40–68 (five+ rounds stale) | 0.5135 | 0.3992 |
| 40–T, uniform | 0.5165 | 0.4123 |
| 55–T | 0.5161 | 0.4098 |
| 40–T, recency half-life 6 rounds | 0.5084 | 0.4121 |

- Including the newest rounds matters (+0.013 on round 74); weighting them more does not. Retrain every round.

### Adopted in v4
- Re-tuned cluster settings with self-training on: +0.0045 / +0.007 / +0.005 on the three unseen sets above.

### No gain (all measured against 0.4838 on unseen rounds 69–73 unless noted)
| Idea | Result |
|---|---|
| Two hash buckets per n-gram | 0.4808 |
| 2,000-char context / word 3-grams / char 3–6-grams / plain char n-grams | 0.4825 / 0.4824 / 0.4813 / 0.4726 |
| Prototype (cluster-centroid) loss | 0.4832 |
| N-gram dropout 0.2 / 0.4 / 0.6 | 0.4766 / 0.4712 / 0.4589 |
| Ridge regression from exact TF-IDF to the smoothed embedding, alone / with self-training | 0.4783 / 0.4850 |
| Second self-training round with 150 classes; reassigning points by the classifier | 0.4844 / 0.4651 |
| Ward / complete / weighted linkage, k-means, spectral, real UMAP + HDBSCAN on the self-trained embedding | 0.42 / 0.465 / 0.469 / 0.468 / 0.43 / 0.45 |
| Smoothing k 15 / 30, α 0.5, 3 iterations (with v4 settings) | within ±0.002 |
| Singleton re-attachment at cosine ≥ 0.95 on top of v4 settings | no additional gain |

- The unquantized table reaches neighbour purity 0.82 on its training rounds and 0.70 on unseen rounds; the
  ternary table 0.69 and 0.66. The sparse table is already the regularised version of the model.

## 2026-10-06 night — literature-driven sweep (test bench `research/lab.py`)

Bench = v4 settings on cached embeddings; set A = rounds 69–73 with a table trained on 40–68, set B = round 74
with a table trained on 40–73. Baseline: A 0.4877, B 0.4174.

### Why nothing on the embedding side moves the score any more
| Embedding → procedure | social (round 73) | arXiv (rounds 69–73) |
|---|---|---|
| Real teacher (all-mpnet-base-v2) → PRESS pipeline | 0.60–0.62 | 0.352 |
| PRESS ternary student → PRESS pipeline | 0.58 | 0.353 |
| Real teacher → UMAP + HDBSCAN | 0.79 | 0.62 |
| 70% teacher / 30% student mix → UMAP + HDBSCAN | 0.70 | — |
| PRESS student → UMAP + HDBSCAN | 0.58 | 0.29–0.34 |

- The smoothing + average-linkage + singleton pipeline is capped near 0.60 even with perfect embeddings, and the
  student is already at ~95% of that cap. Only UMAP + HDBSCAN (which forms the shared noise cluster) goes
  higher, and it needs an embedding much closer to the teacher than any ~77 KB model reaches.
- A one-layer self-attention encoder over hashed words fits training rounds better than the bag model
  (purity 0.846 vs 0.835) but is identical on unseen rounds (0.707 vs 0.707 social; 0.600 vs 0.591 arXiv):
  the limit is what the labelled rounds can teach, not the architecture.

### Methods from the literature, all tested on the bench
| Method (source) | A | B |
|---|---|---|
| Density peaks, 40/80/120 centres; halo points as singletons (Rodriguez & Laio, Science 2014) | 0.474–0.479; halo 0.437–0.451 | 0.390–0.401; 0.338–0.370 |
| Jaccard shared-neighbour graph + Leiden (PhenoGraph, Cell 2015; Traag et al., Sci Rep 2019) | 0.432–0.467 | 0.362–0.394 |
| Hubness reduction: CSLS distance / local scaling / CSLS neighbours (Schnitzer et al., JMLR 2012) | 0.471 / 0.412 / 0.475 | 0.395 / 0.336 / 0.400 |
| Evidence-accumulation consensus over 16 perturbed clusterings (Fred & Jain 2005; SC3, Nat Methods 2017) | 0.481–0.486 | 0.415–0.417 |
| Singletons chosen by ensemble instability | 0.476–0.481 | 0.417–0.419 |
| Shared noise cluster from: ensemble instability, out-of-fold classifier margin (confident learning), small clusters, vote of 4 UMAP + HDBSCAN runs | all below baseline; noise precision 0.19–0.43 | same |
| Ensemble of 2 or 3 independently trained full-size tables | 0.4849 (vs 0.4838 single, v3 settings) | — |
| Noise-isolation loss during training (push ground-truth noise away from everything) | 0.4841 / 0.4797; noise precision 0.33 → 0.34–0.36 | — |

### arXiv knowledge from the public title pool (557k titles from 2025–26; 91% of past competition titles are in it)
| Float table, unseen rounds 69–73 | arXiv purity | teacher-neighbour recall | arXiv score |
|---|---|---|---|
| Baseline | 0.604 | 0.22 | 0.367 |
| + 44 synthetic subsets with teacher-similarity loss | 0.654 | 0.33 | 0.358 |
| + 40% teacher-only batches from the pool | 0.653 | 0.39 | 0.374 |
| arXiv-only specialist, 24,576 × 96 | — | 0.43 | 0.358 (UMAP + HDBSCAN 0.343–0.362) |
| arXiv-only specialist, 65,536 × 192 | — | 0.48 | 0.362 (UMAP + HDBSCAN 0.369–0.375) |

- Distillation from the pool does improve arXiv neighbourhoods, but the pipeline cap hides it, and the gain does
  not survive in the shipped sparse table (pool batches at 15% / 30%: 0.4819 / 0.4760 vs 0.4839).

### Conclusion
v4 stands. No method from this sweep improves it. A real step up needs either an embedding near teacher quality
(not reachable in ~77 KB with any model tried) or a way to predict reference noise (not predictable from our
space by any of ~20 signals).

## 2026-10-07 — two "big step" hypotheses, both rejected

**1. Noise as a learnable property of the text (arXiv).** 110 real teacher-pipeline runs over a fixed pool of
30,000 public 2026 titles (each title seen ~18 times):
- A title that is noise in one sample is noise in another 52% of the time (28% otherwise; base rate 36%).
- Same-cluster pairs stay together 63% of the time.
- Upper bound: the title's true noise frequency over the other runs predicts a held-out run with AUC 0.79;
  precision 0.75 / 0.70 / 0.66 / 0.62 at the top 5 / 10 / 20 / 30%. Only 1.2% of titles are noise in ≥ 90% of runs.
- A shared noise cluster needs ≥ 0.7 precision at ~30% volume, and a packed model predicting this from text
  would sit far below the oracle (text-only detectors reached AUC 0.66–0.70). Rejected.

**2. Context across the round's four calls.** The server process stays alive between `/cluster` calls, so texts
from earlier social subsets could join the neighbour smoothing and self-training of later ones.
| Variant | Set A | Set B | Set C |
|---|---|---|---|
| No context (current) | 0.5310 | 0.4239 | 0.5782 |
| Earlier subsets as context, same k | 0.5236 | 0.4206 | 0.5685 |
| Context with k × 1.5 / k scaled to the pooled size | 0.5234 / 0.5207 | 0.4225 / 0.4214 | 0.5670 / 0.5679 |

Extra points from sibling subsets make every later subset worse: the reference structure is specific to each
5,000-text sample. Rejected.

## 2026-10-06/07 — preparing round 76

Bench baseline (v4 settings): A 0.4877, B 0.4174, C 0.5234. Nothing below improves on v4.

| Tried | Result |
|---|---|
| 19 fine-tuning variants of self-training (classes 60–250, α 3–30, weight 0.5–1.3, training-set rules, 2–3 granularities), density k, singleton share, cluster count | no variant wins on all three sets; all within ±0.003 or worse |
| Low-density points in groups of ~1.5–5, merged by distance, or by a fine cut of the tree, instead of singletons | −0.002 to −0.03; pure singletons are best |
| Boosting the newest round ×5, or the newest three ×3, in training | round 73: 0.5175 vs 0.5203; round 74: 0.4153 / 0.4135 vs 0.4140 |
| Distilling the float table into the sparse table (similarity matching, weight 1 / 3) | 0.4822 / 0.4761 vs 0.4839 |
| Averaging the latent weights over the last 40% of training (decay 0.99 / 0.999) | 0.4811 / 0.4766 vs 0.4839 |

**Recency, measured more precisely (round 74, v4 settings):** trained through round 72: 0.4092; through round 73:
0.4140. Each newer round in training is worth about +0.005 on the next round. Consecutive rounds share almost no
near-duplicate texts (0.4% of round-74 social texts have a cosine ≥ 0.8 match in round 73), so this is topical
drift, not repetition.

**Plan for round 76:** `tools/make_version.sh 5 75` as soon as round 75's data is public (~15:30 UTC).

## 2026-10-07 early — further attempts to beat v4 before round 75's data is public

Run-to-run noise note: the same configuration retrained with a different thread count scored 0.4800 and 0.4839
on set A (v3 settings), so single-run differences below ~0.004 are not evidence.

| Tried | Result |
|---|---|
| Self-training classifier: LinearSVC, logistic regression (logits / probabilities), centroid cosine | all worse than ridge (−0.003 to −0.023) |
| Self-training features: words only; bigger n-gram ranges (320k features); TF-IDF + student embedding | −0.009 to +0.003, mixed signs; +0.0006 / −0.0001 / +0.0013; −0.003 to −0.005 |
| Singletons ranked partly within each cluster, or adjusted by cluster size | −0.002 to −0.019; ±0.002 |
| HDBSCAN directly on the smoothed embedding (noise as singletons) | 0.4816 / 0.4173 vs 0.4877 / 0.4174 — a near tie with a different clusterer |
| Combinations of HDBSCAN and average linkage (product partition, noise AND/OR density, splits) | best 0.4879 / 0.4174 / 0.5262 vs 0.4877 / 0.4174 / 0.5234 — no real gain |
| Char-block weight ×0.5 / ×2; hash split 16,384 + 8,192 or 4,096 + 20,480 | 0.4803 / 0.4791 / 0.4731 / 0.4817 |
| Batch size 3,072 | 0.4835 |
| **New table format:** fixed pseudo-random support, only signs stored (1 bit per weight instead of ~7.4): 26 weights in every row | 0.4766 |
| Same, with 12–96 weights per row allocated by row importance (three allocations) | 0.4750 / 0.4757 / 0.4796 |

- **Bug fixed in `research/train.py`:** the saved column scales omitted the per-column mean |W| factor that the
  training forward pass applies. Columns differ by at most 13%, and with identical training the fixed table scores
  0.4805 vs 0.4800, so v1–v4 lost essentially nothing; future tables are saved correctly.
- Three structurally different final stages (average linkage, HDBSCAN on the smoothed space, consensus) and two
  table formats all land on the same score. That is the ceiling of this embedding under this family of pipelines.

## 2026-10-07 morning — larger bench (rounds 61–74 unseen, table trained on 40–60) and v4.1

### Joint random search of all pipeline settings (odd rounds to search, even rounds to confirm)
- Social (21 + 21 subsets): v4's settings are the best of 71 configurations on the search half (0.4689) and are not
  beaten on the confirmation half.
- arXiv (7 + 7 subsets, then 160 configurations on all 14): fewer final clusters is better. Marginal effect of the
  cluster count: 80 → +0.0054, 120 → +0.0019, 180 (v4) → −0.0017, 260 → −0.0055.
- **Adopted as v4.1:** arXiv 80 clusters, singleton share 0.30. 14 unseen arXiv subsets: 0.3624 vs 0.3561 (odd
  +0.0055, even +0.0070, 9 of 14 better). End to end: rounds 69–73 0.4874 vs 0.4870; round 73 0.5247 vs 0.5233;
  round 74 0.4180 vs 0.4171.
- Singleton re-attachment at cosine 0.90–0.97 on 56 subsets: +0.0001 to −0.0053. Rejected for good.

### Training options on the larger bench (v4.1 pipeline; three seeds of the current recipe: 0.4323 / 0.4336 / 0.4355)
| Option | Round score |
|---|---|
| x/y/z loss weight 1.0 | 0.4322 |
| No sparsity annealing | 0.4338 |
| 40,960 × 64 table | 0.4335 |
| 8 row-scale levels | 0.4319 |
| Start training at round 30 / 20 (two seeds) / 10 / 1 instead of 40 | 0.4346 / 0.4381, 0.4376 / 0.4356 / 0.4355 |

- Starting at round 20 looked like a real gain here (+0.004), but it did not survive the realistic check: tables
  trained through round 68 and tested on rounds 69–74 score 0.4720 / 0.4704 from round 20 against 0.4721 / 0.4761 /
  0.4728 from round 40. Older rounds only help when the training window is short. Not adopted.

## 2026-10-07 08:00 UTC — per-batch adaptation and learned merging (14-round bench)

| Question | Social (42 subsets) | arXiv (14 subsets) |
|---|---|---|
| Current settings | 0.4557 | 0.3614 |
| Best single (clusters, singleton share) for all subsets | 0.4571 | 0.3623 |
| Oracle: best (clusters, singleton share) per subset | 0.4640 | 0.3652 |
| Oracle: fine pieces of the tree merged by their majority ground-truth label (300 / 600 fine clusters) | 0.4939 / 0.5032 | 0.3774 / 0.3787 |

- Per-batch setting choice is worth at most +0.008 and the best predictors (relative density, first-neighbour
  distance) correlate only 0.3–0.4 with the per-subset optimum. Not pursued.
- *Which pieces are merged* is worth up to +0.047 on social, so a merge rule was learned from past rounds:
  pairs of fine clusters described by 14 features (centroid cosine in the smoothed / raw / classifier / TF-IDF
  spaces, cross-neighbour links, mean pairwise similarity, cophenetic distance, sizes, mutual ranks, tightness);
  trained on rounds 61–67, tested on 68–74.

| Merge model (test rounds 68–74, social) | Pair AUC | Score |
|---|---|---|
| Uniform tree cut (current) | — | 0.4720 |
| Gradient boosting, threshold chosen on training rounds | 0.708 | 0.4666 |
| Logistic regression | 0.740 | 0.4658 |
| Oracle pair labels | 1.0 | 0.5092 |

arXiv: 0.3504 / 0.3665 vs 0.3685. The pair model cannot tell which pieces belong together any better than the
average-linkage tree already does; rejected.

## 2026-10-07 — multi-agent campaign (35 agents: 8 scouts, 3 judges, experimenters, adversarial verifiers, integrators)

Bench: `research/bench.py`; brief: `docs/AGENT_BRIEF.md`; agent work in `research/wf/` (not committed).
Deltas are vs v4.1. D = rounds 61–74 (table 40–60), A = 69–73 (table 40–68), B = 74 (40–73), C = 73 (40–72).

| Idea | D (odd / even) | A / B / C | Verdict |
|---|---|---|---|
| **Learned singleton ranker** (17 features; logistic fit to the first-order score gain of abstaining each point) | +0.0028 (+0.0027 / +0.0028) | +0.0020 / +0.0029 / +0.0020 | survived; end to end +0.0024 / +0.0021 / +0.0025 |
| **Ranker with 2 lexical features** (10-NN agreement and centroid margin in exact TF-IDF space; 19 features) | +0.0048 (+0.0047 / +0.0049) | +0.0032 / +0.0037 / +0.0027 | survived; end to end +0.0034 / +0.0031 / +0.0033 |
| **Language split** of mixed-language clusters | +0.0037 (+0.0033 / +0.0041) | +0.0005 / 0 / 0 | survived; insurance (pays when new language communities appear, e.g. round 66 +0.024) |
| Link-post rule (`preview.redd.it` captions grouped when ≥ 20) | +0.0011 | +0.0011 / 0 / +0.0004 | small but consistent; kept in the stack |
| Image-link seeded pseudo-class | +0.0024 | +0.0015 / 0 / +0.0008 | refuted (loses on earlier periods where it fires) |
| **Full stack** (ranker + language split + link rule) | +0.0076 (+0.0072 / +0.0081), 51 of 56 better | +0.0036 / +0.0029 / +0.0028 | survived; end to end +0.0038 / +0.0028 / +0.0025 |
| Compact container (ANS-coded table + minified code) | lossless; 46.7k chars vs 49.1k | — | adopted (frees 3,300 chars) |
| Denser table using the freed space (nz 0.036, 0.046, 0.050; 19-bit characters) | +0.0007 to +0.0026, one or two seeds | through-68 check +0.0016 / +0.0017 | below the bar; 19-bit characters also untested on the server |
| Pair-count-weighted contrastive loss | −0.0014 | — | no gain |
| Self-training set chosen by density and lexical agreement | +0.0018 on one table, +0.0004 / +0.0005 on two others | +0.0011 / +0.0008 / +0.0005 | inconclusive |
| Lexical-fused smoothing graph | +0.005 | ≈ 0 | stale-table effect only; rejected |
| Removing re-served round files from training | −0.0021 | — | no gain (bench now aware of the re-serve) |
| Re-tuning cluster count / singleton share on fresh tables (55 subsets) | +0.0001 | — | no gain |

**Shipped:** v4.2 = stack with the 17-feature ranker; v4.3 = stack with the 19-feature ranker (47,009 chars).

**Not adopted, needs the user's decision:** a scout found that the organisers have re-served byte-identical
datasets (round 54 = round 43, round 55 = round 44, round 65 subset 3 = round 43 subset 3). A file could recognise a
re-served batch and return its stored public labels. That exploits a flaw in the evaluation rather than improving
the method, and contradicts the competition's stated intent ("memorising revealed rounds does not help").

## 2026-10-07 15:40 UTC — round 75 (unseen by the production table) and v5

| File | Subset 1 | Subset 2 | Subset 3 | arXiv | Round |
|---|---|---|---|---|---|
| v4.1 | 0.5156 | 0.4796 | 0.4731 | 0.3368 | 0.4513 |
| v4.2 | 0.5089 | 0.4810 | 0.4738 | 0.3452 | 0.4523 |
| v4.3 | 0.5109 | 0.4843 | 0.4732 | 0.3461 | 0.4536 |
| Official top of round 75 | | | | | 0.4518 |

- The stack gained +0.0023 here, a little under the bench estimate (+0.003). Subset 1 lost 0.005–0.007: the ranker
  lost there and gained on the other three subsets; the link rule and language split each moved subsets by
  ±0.002 (net −0.0005 and +0.0007 on this round). One round does not overturn the bench evidence; kept.
- v4.1 (= v4 code) scores 0.4513, level with the copies that led round 75 (0.4513–0.4518).
- **v5** = v4.3 code + table retrained on rounds 40–75 (`tools/make_version.sh 5 75 19`).
