"""Calibrate UMAP/HDBSCAN settings against real arXiv-subset ground truth (teacher embeddings computed here)."""
import glob, os, sys, numpy as np, pandas as pd, warnings, torch
warnings.filterwarnings("ignore")
import umap, hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
torch.set_num_threads(4)
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
res = {}
for f in [glob.glob(f"data/round_{r:04d}/*arxiv*.parquet")[0] for r in (62, 65, 68, 70, 72, 73)]:
    df = pd.read_parquet(f); gt = df.label.values
    c = "cache/" + os.path.basename(f).replace(".parquet", ".mpnet.npy")
    if os.path.exists(c): E = np.load(c)
    else: E = m.encode(df.text.tolist(), batch_size=128, show_progress_bar=False); np.save(c, E)
    for nn in (15,):
        for seed in (42, 7):
            U = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric="cosine", random_state=seed, n_jobs=1).fit_transform(E)
            for ms in (1, 3, 5, 8, 10, 15, 25):
                for meth in ("eom", "leaf"):
                    l = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=ms, cluster_selection_method=meth).fit_predict(U)
                    res.setdefault((nn, ms, meth), []).append(((max(0, ari(gt, l)) + nmi(gt, l)) / 2, l.max() + 1, (l == -1).mean(), gt.max() + 1, (gt == -1).mean()))
    print(os.path.basename(f), flush=True)
for k, v in sorted(res.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]])):
    v = np.array(v).mean(0); print(k, "score %.4f  k %.1f (gt %.1f)  noise %.3f (gt %.3f)" % tuple(v))
