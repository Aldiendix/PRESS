# PRESS

Solutions for the **Text Clustering** competition on [Apex](https://apex.macrocosmos.ai/competitions/10)
(Bittensor Subnet 1, competition 10).

The task: cluster ~5,000 raw texts (X/Reddit posts or arXiv titles) so that the result matches a hidden ground
truth built with `all-mpnet-base-v2` embeddings → UMAP → HDBSCAN. The submission is one Python file of at most
50,000 characters that runs on CPU with only numpy, scipy and scikit-learn, without internet, in 90 seconds.
Score per subset is `(max(0, ARI) + NMI) / 2`; a round is the mean of four subsets.

## How the solution works

1. **Hashed n-grams.** Each text becomes counts of word 1–2-grams and character 3–5-grams, hashed into fixed
   buckets, with `log1p` and batch IDF weighting.
2. **Packed student table.** A sparse ternary matrix (weights in {−1, 0, +1} with per-column and per-row scales)
   maps those buckets to an embedding. It is trained offline, with quantization in the loop, to pull together
   texts that share a ground-truth cluster in past rounds. The table is LZMA-compressed and stored in the source
   file as text, 15 bits per character.
3. **Neighbour smoothing.** Every embedding is mixed with the mean of its nearest neighbours a few times.
4. **Clustering.** Average-linkage agglomerative clustering on cosine distance.
5. **Noise.** The lowest-density points are returned as singleton clusters.

## Layout

| Path | Purpose |
|---|---|
| `versions/vN/solution.py` | the built single-file submission for version N |
| `versions/vN/NOTES.md` | what changed and how it scored |
| `src/solution_template.py` | readable source of the miner (table inserted at build time) |
| `tools/fetch.py` | download round data, leaderboards and revealed code from the public dashboard API |
| `tools/evaluate.py` | local replica of the official scorer (matches official per-subset scores exactly) |
| `tools/build.py`, `tools/pack.py` | pack a trained table into the template and check the character limit |
| `research/` | training and experiment scripts |
| `docs/` | analysis of revealed solutions and the research log |

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy==2.3.5 scikit-learn==1.6.1 fastapi==0.124.2 uvicorn==0.38.0 pydantic==2.12.5 pyarrow pandas
.venv/bin/python tools/fetch.py 71 72 73                 # round data + top submissions
.venv/bin/python tools/evaluate.py versions/v1/solution.py --rounds 71 72 73
```

Training the table needs a second environment with PyTorch (CPU is enough); see `research/train.py`.
