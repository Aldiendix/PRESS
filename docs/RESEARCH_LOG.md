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
