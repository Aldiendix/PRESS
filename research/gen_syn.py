"""Synthetic arXiv subsets: sample 5000 pool titles, label them with the teacher pipeline (UMAP + HDBSCAN).

Written as data/round_9NNN/round_9NNN_subset_arxiv_syn.parquet (+ cached teacher embeddings) so the normal
training code can read them.  Usage: gen_syn.py <first> <count> <rows_available>
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
first, count, avail = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
P = pd.read_parquet("cache/arxiv_pool.parquet").iloc[:avail]
recent = np.flatnonzero(P.id.str[:2].values == "26"); older = np.flatnonzero(P.id.str[:2].values == "25")
def one(i):
    import umap, hdbscan
    r = first + i; d = f"data/round_{r:04d}"; name = f"round_{r:04d}_subset_arxiv_syn{i:05d}"
    if os.path.exists(f"{d}/{name}.parquet"): return r
    rng = np.random.default_rng(r); n26 = int(5000 * 0.82)
    idx = np.concatenate([rng.choice(recent, n26, replace=False), rng.choice(older, 5000 - n26, replace=False)]); rng.shuffle(idx)
    E = np.load("cache/arxiv_pool.f16.npy", mmap_mode="r")[np.sort(idx)].astype(np.float32); idx = np.sort(idx)
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=r, n_jobs=1).fit_transform(E)
    lab = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=int(rng.choice([5, 8, 8]))).fit_predict(U)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(dict(text=P.title.values[idx], label=lab.astype(np.int32), x=U[:, 0], y=U[:, 1], z=U[:, 2])).to_parquet(f"{d}/{name}.parquet")
    np.save(f"cache/{name}.mpnet.npy", E)
    return r, int(lab.max() + 1), float((lab == -1).mean())
if __name__ == "__main__":
    with ProcessPoolExecutor(int(os.environ.get("J", "8"))) as ex:
        for o in ex.map(one, range(count)): print(o, flush=True)
