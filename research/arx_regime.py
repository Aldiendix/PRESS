import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
import umap, hdbscan
from train import load_subset, files_for
from grid2 import fixz
from lab import smooth, sc
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
def mine(Z, S=180, fr=.35, k=35):
    Zs = smooth(Z, k); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; l = fcluster(linkage(Zs, "average", "cosine"), S, "maxclust"); o = np.argsort(-d)[: int(len(Z) * fr)]; l[o] = 10**6 + np.arange(len(o)); return l
def uh(Z, ms, single, seed=1, smooth_k=0):
    if smooth_k: Z = smooth(Z, smooth_k)
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed, n_jobs=1).fit_transform(Z); l = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=ms).fit_predict(U)
    if single: m = l == -1; l[m] = 10**6 + np.arange(m.sum())
    return l
def one(f):
    s = load_subset(f, 8192, 16384, True); gt = s["y"]; res = {}
    T = normalize(np.load("cache/" + os.path.basename(f).replace(".parquet", ".mpnet.npy")))
    for nm, Z in (("teacher", T), ("float+pool", fixz(s["X"] @ np.load("cache/W_fp4.npy"))), ("float", fixz(s["X"] @ np.load("cache/W_float96.npy"))), ("ternary v4-like", fixz(s["X"] @ np.load("cache/W_w2.npy")))):
        res[nm + " | my pipeline"] = sc(gt, mine(Z)); res[nm + " | umap+hdbscan noise=-1"] = sc(gt, uh(Z, 8, False)); res[nm + " | umap+hdbscan noise=singletons"] = sc(gt, uh(Z, 8, True))
        res[nm + " | smooth then umap+hdbscan -1"] = sc(gt, uh(Z, 8, False, smooth_k=15))
    return res
if __name__ == "__main__":
    fs = [f for f in files_for([69, 70, 71, 72, 73]) if "arxiv" in f]
    with ProcessPoolExecutor(5) as ex: R = list(ex.map(one, fs))
    for k in R[0]: print("%-48s %.4f" % (k, np.mean([r[k] for r in R])))
