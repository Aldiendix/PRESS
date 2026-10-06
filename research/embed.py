"""Cache teacher (all-mpnet-base-v2) embeddings for round parquet files -> cache/<name>.npy"""
import glob, os, sys, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(int(os.environ.get("NT", "16")))
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
files = [f for r in sys.argv[1:] for f in sorted(glob.glob(f"data/round_{int(r):04d}/*.parquet"))]
for f in files:
    out = "cache/" + os.path.basename(f).replace(".parquet", ".mpnet.npy")
    if os.path.exists(out):
        continue
    t = pd.read_parquet(f).text.tolist()
    order = np.argsort([len(x) for x in t])
    e = m.encode([t[i] for i in order], batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    emb = np.empty_like(e); emb[order] = e
    np.save(out, emb.astype(np.float32)); print(out, emb.shape, flush=True)
