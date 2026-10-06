"""Teacher embeddings for selected subsets. Usage: embed_sel.py <arxiv|social|all> <round> [<round> ...]"""
import glob, os, sys, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(int(os.environ.get("NT", "14")))
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu"); kind = sys.argv[1]
for r in sys.argv[2:]:
    for f in sorted(glob.glob(f"data/round_{int(r):04d}/*.parquet")):
        arx = "arxiv" in f or "subset_4" in f
        if (kind == "arxiv" and not arx) or (kind == "social" and arx): continue
        out = "cache/" + os.path.basename(f).replace(".parquet", ".mpnet.npy")
        if os.path.exists(out): continue
        t = pd.read_parquet(f).text.tolist(); order = np.argsort([len(x) for x in t])
        e = m.encode([t[i] for i in order], batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        emb = np.empty_like(e); emb[order] = e; np.save(out, emb.astype(np.float32)); print(out, flush=True)
