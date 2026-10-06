"""Teacher (all-mpnet-base-v2) embeddings for the recent arXiv title pool -> cache/arxiv_pool.{parquet,f16.npy}"""
import os, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(int(os.environ.get("NT", "12")))
p = pd.read_parquet("data/arxiv/titles_2023plus.parquet"); p = p[p.id.str[:2].isin(["25", "26"])].reset_index(drop=True)
p.to_parquet("cache/arxiv_pool.parquet"); print(len(p), flush=True)
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
out = np.lib.format.open_memmap("cache/arxiv_pool.f16.npy", mode="w+", dtype=np.float16, shape=(len(p), 768))
import time; t0 = time.time(); B = 20000
for i in range(0, len(p), B):
    out[i:i + B] = m.encode(p.title.iloc[i:i + B].tolist(), batch_size=128, show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
    out.flush(); print(i + B, "%.0fs" % (time.time() - t0), flush=True)
open("cache/arxiv_pool.done", "w").write("ok")
