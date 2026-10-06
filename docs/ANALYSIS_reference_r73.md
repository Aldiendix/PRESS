# Analysis of the best revealed solution (round 73, `5C55Guoe` v1, official score 0.4749)

File: `reference/r73_5C55Guoe_v1_0.4749.py` — 419 lines, 49,490 characters (112 KB as UTF-8).
The source is minified (one-letter names). Line numbers below refer to that file.
Local replay with `tools/evaluate.py` reproduces the official round-73 score exactly (0.4749).

The current leader's file (`5Eh8oDoA` v3, 0.4949 on round 73) is revealed on 2026-10-06 06:43 UTC and is
not covered here; the 14 revealed files from rounds 71–73 are all ~112 KB and share this structure.

## 1. How it fits a model into 50,000 characters

The limit counts *characters*, not bytes. The file stores binary data as CJK/Hangul characters, 15 bits each:

| Lines | Name | What it is |
|---|---|---|
| 67 | `Ak` (12,334 chars ≈ 23 KB) | LZMA-compressed table for **social posts**: 12,288 hashed n-gram rows × 48 dims, 2 bits per weight, plus a 4-level codebook per dimension |
| 217 | `BD` (19,092 chars ≈ 36 KB) | LZMA-compressed table for **arXiv titles**: sparse list of non-empty rows, 4 bits per weight, per-dimension scale |
| 202–207, 268–271 | `B2`, `BN` | decoders: each character → `ord(c) - 0x4E00` → 15 bits appended to a bit stream |
| 156–165, 272 | `Ax`, `BO` | unpack the tables into float32 matrices |

Both tables map hashed n-grams to an embedding: `HashingVectorizer` word 1–2-grams (8,192 buckets) and
`char_wb` 3–5-grams (4,096 buckets) → `log1p` → L2-normalise → multiply by the table → L2-normalise
(`Ay` lines 166–173 for social, `BP` lines 273–279 for titles). These are students distilled from a
sentence-embedding teacher; they are the only "pretrained knowledge" in the file.

## 2. Request flow (`cluster_texts`, lines 342–409)

1. **Clean** every text (`Ar`, line 76): lowercase, strip URLs/@mentions/punctuation. Empty results get label `-1`.
2. **Huge batches** (> 15,000 texts, lines 349–358): TF-IDF → SVD(128) → MiniBatchKMeans. Not used at 5,000 texts.
3. **Detect the arXiv subset** (line 359): median length < 130 chars, < 8% of texts contain a newline,
   < 3% contain a link → flag `N`. If set, run the title pipeline `BV` and return.
4. Otherwise run the **social pipeline** (lines 363–405).

## 3. Social pipeline

**Feature views, concatenated with weights (lines 363–375):**

| View | Function | Detail |
|---|---|---|
| TF-IDF LSA | `t` (143–155) | word 1–2-grams + `char_wb` 3–5-grams, sublinear tf, `max_df=0.5` → SVD(192) |
| Batch word vectors | `Aw` (122–142) | word co-occurrence in a ±10 window → PPMI → SVD(128) → documents as TF-IDF-weighted sums |
| Distilled table | `Ay` (166–173) | the packed student described above |

- `Av` (112–115) measures how much variance the first principal component explains; `s` (116–121) removes the
  top 1–4 principal components ("all-but-the-top"). How many are removed, and the view weights, depend on that
  variance and on the median text length.
- A special case fires when the median length is between 230 and 330 characters (different weights, cluster
  count fixed at 28, noise share 0.25). This looks tuned to particular past rounds.

**Graph smoothing (lines 376–379):**
- `Az` (174–182): each vector is replaced by 0.8·itself + 0.2·mean of its 12 nearest neighbours, twice.
- `A_` (183–187): 15-NN graph → normalised adjacency → top eigenvectors (26 by default) appended as extra features.

**Clustering (lines 381–392):**
- `B0` (188) computes "relative density" = mean 14th-neighbour distance ÷ mean random-pair distance.
- `B1` (189) maps that to a cluster count between 40 and 220 and a "noise" share between 0.25 and 0.40.
- Average-linkage agglomerative clustering on cosine distance, cut at that cluster count.

**Post-processing (lines 393–404):**
- `A4`/`A5` (78–95): detect non-Latin scripts per text (Cyrillic, CJK, Arabic, Hebrew, Thai, Devanagari) and merge
  clusters dominated by one script — the teacher groups same-language posts.
- `Au` (105–111): if the first principal component is weak, merge the two clusters whose TF-IDF centroids are most similar (≥ 0.7).
- **"Noise" as singletons (397–402):** the 25–40% of points with the largest k-NN distance each get their own
  unique cluster id (not `-1`). `At` (96–104) protects up to 3 large, tight clusters from this.
- `RED` (318–326): a singleton or tiny cluster (< 5) rejoins a real cluster if its cosine similarity to that centroid is ≥ 0.77.

## 4. arXiv-title pipeline (`BV`, lines 293–317)

- Views: BM25-weighted word + char n-grams → SVD(192) (`BL`, `AD`); PPMI word vectors from within-title
  co-occurrence → SVD(96) (`BM`); the title table (`BP`), weighted ×3.
- k-NN smoothing with k = 35, α = 0.4, 4 iterations (`BT`).
- Average-linkage clustering cut at 100 clusters; the 30% of points with the largest 12th-neighbour distance become singletons.

## 5. What the local measurements say

All numbers are on round 73 (4 subsets), using the official ground truth in the round parquet files.

| Experiment | Score |
|---|---|
| This solution as submitted | 0.475 |
| Its features + real UMAP + HDBSCAN instead of its clustering | 0.43 |
| Real teacher embeddings (all-mpnet-base-v2) + its style of agglomerative clustering | 0.41–0.45 |
| Real teacher embeddings + UMAP(15 nn, 5 dims) + HDBSCAN(min size 30, min samples 5), different seed | 0.76 |

- **Ceiling.** Re-running the teacher pipeline with another UMAP seed scores ~0.75, so that is the practical
  maximum; the field is at ~0.47–0.49 on this round.
- **Embedding quality is the bottleneck.** Its features recover only ~16% of the teacher's 15 nearest neighbours
  on the arXiv subset.
- **Noise detection is weak.** For the round-73 leader's predictions, only ~30% of the points turned into
  singletons are true noise (base rate 10–26%). With perfect noise labels that submission would score 0.74 instead of 0.49.
- **Why singletons instead of `-1`.** With precision this low, one shared noise cluster creates many wrong pairs;
  relabelling the leader's singletons as `-1` lowers its score by 0.01–0.05 per subset.
- **What noise really is.** Ground-truth noise is poorly predicted by density in teacher space (AUC 0.57–0.78)
  but well predicted by neighbour-label disagreement (AUC 0.88–0.94): noise points sit *between* clusters.
- **Ground-truth facts.** Every subset has 5,000 texts; the smallest cluster is always ≥ 25 (HDBSCAN
  `min_cluster_size=25`); recent rounds have 23–49 clusters and 10–30% noise.
- **Leakage caution.** Reference files score higher on rounds before their submission date than after, which
  suggests their tables were trained on those rounds' public labels. Compare models only on rounds none of them saw.
