# PRESS agent brief

Read this fully before doing anything. Project root: `/root/PRESS`. Always run commands from that directory.

## 1. The task

- Competition: Apex (Bittensor subnet 1) "Text Clustering". Each round has 4 batches ("subsets") of 5,000 texts:
  3 of social posts (X / Reddit, mostly English, >= ~80 characters) and 1 of arXiv paper titles.
- The solution receives the raw texts of one batch and returns one integer cluster id per text.
- Hidden reference for each batch: `all-mpnet-base-v2` sentence embeddings -> UMAP (15 neighbours, 5 dims,
  min_dist 0, cosine) -> HDBSCAN (min_cluster_size 25, min_samples about 5-8, EOM). It has 20-49 clusters and labels
  10-30% of the points as noise (`-1`). **Noise counts as one ordinary cluster in scoring.**
- Score of a batch = `(max(0, ARI) + NMI) / 2` (scikit-learn, over all 5,000 points). Round score = mean of the 4 batches.
  A new entry must beat the current top by 1% (relative) to take first place, so +0.004 to +0.005 matters.

### Hard runtime constraints (the deployed solution)
- One Python file of at most **50,000 characters** (Python `len()` of the source text).
- Python 3.12 with ONLY `numpy 2.3.5`, `scipy`, `scikit-learn 1.6.1`, the standard library (plus fastapi / uvicorn /
  pydantic for the HTTP server). No other packages, no network, no data files.
- CPU only (assume one core), **90 s** per batch, **1.5 GB** RAM.
- No pretrained model except what is packed inside the file itself (currently a table of about 77 KB, stored as
  ~42,000 characters at 15 bits per character). Offline you may use anything (PyTorch, sentence-transformers, UMAP,
  hdbscan are installed in `.venv-research`), but the shipped method must run under the constraints above.

## 2. The current solution (v4.1)

`src/solution_template.py` is the readable source; `tools/build.py` packs a trained table into it;
`versions/v4.1/solution.py` is the built file (49,111 characters, ~5 s and ~530 MB per batch).

1. **Features.** Lower-cased text with URLs / @mentions / punctuation removed, cut at 1,000 chars. Hashed word
   1-2-grams (8,192 buckets) and character 3-5-grams within word boundaries (16,384 buckets); `log1p` counts,
   batch IDF, L2 normalisation.
2. **Student table.** A 24,576 x 96 sparse ternary matrix (3.3% non-zero, per-column and per-row scales) maps
   features to a 96-d embedding `Z`. Trained offline (`research/train.py`) with quantisation in the loop:
   supervised contrastive loss on the reference labels of past rounds (batches drawn from one subset at a time,
   noise points are negatives only) plus a neighbour-matching loss on the published `x, y, z` coordinates.
3. **Smoothing.** 4 rounds of `Z <- normalize(0.6 Z + 0.4 mean of k nearest neighbours)` (k = 20 social, 35 arXiv).
4. **First pass.** Average-linkage agglomerative clustering on cosine distance.
5. **Self-training.** A ridge classifier (alpha 10) on the batch's exact TF-IDF matrix (word 1-2-grams + char
   3-5-grams, no hashing) learns 90 first-pass clusters from the dense points; its centred, normalised decision
   scores are appended to `Z` (weight 0.75); steps 3-4 are repeated.
6. **Output.** Tree cut at 120 clusters (social) / 80 (arXiv); the 20% (social) / 30% (arXiv) lowest-density points
   (largest cosine distance to the 15th neighbour) each become a singleton cluster.

## 3. Structural findings so far (measured, see `docs/RESEARCH_LOG.md`)

- **Noise is the main loss and it is not recoverable.** With the current clusters and perfect knowledge of which
  points are reference noise, the score would be 0.75 instead of 0.49. But a shared noise cluster only pays off at
  >= 0.7 precision at ~30% volume, and every signal tried (about 25: densities, margins, ensemble instability,
  votes of repeated UMAP+HDBSCAN runs on our embedding, supervised detectors) reaches ~0.3. Even re-running the
  real teacher pipeline agrees with the reference noise at only ~0.7 precision.
- **Singletons work as abstention**, not noise detection: 73% of them are noise or misassigned points.
- **The pipeline is capped.** The real teacher embeddings pushed through this pipeline score only 0.60 on social
  subsets and 0.35 on arXiv; the student already gets 0.58 / 0.35 there. Only UMAP + HDBSCAN goes higher
  (0.79 / 0.62 on teacher embeddings) and it needs embeddings far closer to the teacher than the student is
  (teacher-neighbour recall@15 of the student is about 0.2).
- **Merge structure has headroom but no known signal reaches it.** Merging fine tree pieces by their majority
  reference label would give +0.047 on social subsets; a pair model with 14 features only reaches AUC 0.74 and
  scores below the plain tree cut.
- **Capacity curve** (score vs table size): 47 KB 0.476, 77 KB 0.484, 128 KB 0.487, 207 KB 0.491, float 0.493.
- **Recency:** each newer round included in training is worth about +0.005 on the next round.
- **Seed noise:** three seeds of the same training recipe score 0.4324 / 0.4338 / 0.4356 on bench set D
  (mean 0.4339, sd 0.0016). Pipeline-only changes are evaluated on fixed tables and are deterministic.

## 4. Already tried without gain - do NOT propose or repeat these

Details and numbers are in `docs/RESEARCH_LOG.md` (read it). Summary:

- *Table / training:* bigger, denser, wider tables; float table; 128 dims; MLP head; one-layer self-attention
  encoder; temperature 0.07 / 0.15; prototype loss; n-gram dropout; weight averaging (EMA); batch 3,072; 15k steps;
  smoothing inside the loss; noise-isolation loss; teacher-embedding distillation; distilling the float table into
  the sparse one; synthetic arXiv subsets and teacher-only batches from a 557k public arXiv title pool; recency
  weighting / boosting the newest rounds; starting from rounds 1 / 10 / 20 / 30 instead of 40; double hashing;
  2,000-char context; word 3-grams; char 3-6-grams; plain char n-grams; URL / mention / punctuation tokens kept;
  char-block weight x0.5 / x2; other hash splits; product quantisation; fixed-support binary format (uniform and
  row-adaptive); ensembles of 2-3 tables; arXiv-only table.
- *Pipeline:* extra views (TF-IDF LSA, PPMI / retrofitted word vectors); removing principal components; real UMAP
  + HDBSCAN on our embedding; HDBSCAN on the smoothed embedding (a tie) and its hybrids with average linkage;
  Ward / complete / weighted linkage; k-means; spectral clustering; density peaks (incl. halo); Leiden on a
  Jaccard shared-neighbour graph; hubness reduction (CSLS, local scaling); consensus over perturbed clusterings;
  similarity-weighted / mutual-kNN / dynamic / shared-neighbour smoothing; per-batch adaptive cluster count or
  singleton share; learned abstention ranker; cluster-relative or size-aware singleton choice; grouping low-density
  points instead of singletons; re-attaching singletons to centroids; re-splitting large clusters; learned merging
  of tree pieces; context from earlier batches of the same round.
- *Self-training:* 2-3 iterations; 40-250 classes; alpha 3-30; weight 0.5-1.3; LinearSVC / logistic / centroid
  classifiers; words only; bigger n-gram ranges; TF-IDF + student embedding as classifier input; ridge regression
  to the smoothed embedding; reassigning points by the classifier; two or three granularities at once.
- *Settings:* a joint random search of all pipeline settings on 14 unseen rounds (v4's social settings were the
  best of 71; arXiv moved to 80 clusters / 0.30 in v4.1).

## 5. The bench (`research/bench.py`)

Environment: `/root/PRESS/.venv-research/bin/python` (PyTorch, sentence-transformers, umap-learn, hdbscan, igraph,
scikit-learn). Set `OMP_NUM_THREADS=1` for bench runs; parallelism comes from processes (`jobs=`).

| Set | Test rounds | Subsets | Table trained on | Baseline (v4.1) |
|---|---|---|---|---|
| **D** (main) | 61-74 | 56 | rounds 40-60 | 0.4324 (social 0.4557, arXiv 0.3625) |
| A | 69-73 | 20 | rounds 40-68 | 0.4885 |
| B | 74 | 4 | rounds 40-73 | 0.4183 |
| C | 73 | 4 | rounds 40-72 | 0.5248 |

A / B / C use texts that are also in D, with fresher tables (closer to deployment). Every score is on rounds the
table never saw.

```python
import sys; sys.path.insert(0, "research")
import numpy as np, bench

def my_variant(ctx):                         # ctx = one subset: f, rnd, arx, gt, Z (5000x96), X (exact TF-IDF)
    r = bench.pipeline(ctx, detail=True)      # v4.1 with all intermediates: labels, clusters, singletons, Zs, d,
    labels = r["labels"].copy()               #   link, ZA, P, pseudo, train_mask, Zs1, d1, link1
    ...                                       # your change. NEVER use ctx["gt"] inside the method.
    return labels                             # or a dict {"name": labels, ...} to test several variants at once

if __name__ == "__main__":
    bench.compare(my_variant, sets=("D",), jobs=4)      # paired vs baseline; prints delta, odd / even rounds, wins
```

- `bench.pipeline(ctx, settings={...}, refine={...})` overrides settings (`k, alpha, iters, clusters, share`;
  `classes, ridge_alpha, weight`). `bench.texts(ctx)` / `bench.texts(ctx, raw=True)` give the texts.
  `bench.score(gt, labels)` is the official metric. `bench.smooth`, `bench.density` are the building blocks.
- **Training-side ideas:** train a table on rounds 40-60 ONLY, then score it on rounds 61-74:
  ```bash
  OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw 8192 --fc 16384 --d 96 --nz 0.033 \
     --steps 6000 --idf 1 --wx 0.5 --rows 4 --nz_start 0.3 --anneal 0.5 --train 40 60 --out cache/wf_<id>_s0.npy
  OMP_NUM_THREADS=1 .venv-research/bin/python research/bench.py cache/wf_<id>_s0.npy      # prints the round score
  ```
  That is the current recipe (about 15 minutes at 4 threads); `--seed 1` gives another seed. Baseline seeds on D:
  0.4324, 0.4338, 0.4356. If you need a modified trainer, copy `research/train.py` into your own directory.
  Hashed features are cached in `cache/feat/` (see `research/feats.py`); ground-truth round files are
  `data/round_NNNN/*.parquet` with columns `text, label, x, y, z`.
- End-to-end check of a built single file: `.venv/bin/python tools/evaluate.py <solution.py> --rounds 73 74 --jobs 4`
  (`.venv` has exactly the sandbox package versions).

## 6. Evidence standard

- Pipeline changes: report `bench.compare` on **D**. A change counts as a candidate win only if the mean delta is
  >= +0.002 AND the delta is positive on both the odd-round and the even-round half. If you tuned any parameter,
  tune it on odd rounds and report the even-round number as the honest one. Then also report A, B and C.
- Training changes: two seeds on D; a candidate win needs a mean >= 0.4375 (baseline mean 0.4339, sd 0.0016).
- The method must not read `ctx["gt"]` (analysis may), must run with numpy / scipy / scikit-learn only, and must fit
  the time and memory limits (report seconds per subset on one core).
- Report negative results plainly. A truthful "no gain" is a useful result; an optimistic claim that does not
  reproduce wastes a paid submission.

## 7. Rules (strict)

- Write ONLY inside your own directory `research/wf/<your-id>/` and cache files named `cache/wf_<your-id>_*`.
  Do not modify, move or delete any existing file (in particular `research/*.py`, `src/`, `tools/`, `versions/`,
  `docs/`, `cache/` files you did not create). Copy a file into your directory if you need to change it.
- CPU: at most 4 processes or threads at any time (`jobs=4`, `--threads 4`, `OMP_NUM_THREADS` accordingly).
  Other agents share this 16-core machine.
- Never kill processes you did not start. Do not use `pkill` / `killall`.
- No git commits or pushes, no network submissions to the competition, do not read `/root/.env` or anything under
  `/root/.bittensor`. Web access is for reading documentation and papers only.
- Keep command output short (pipe through `tail` / `grep`); do not print large arrays.
