"""Evaluate student table(s) on held-out rounds with a fixed simple clusterer + kNN diagnostics."""
import sys, glob, os, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "research")
from train import load_subset, files_for
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
import os as _os
IDF = int(_os.environ.get("IDF", "0"))
sc = lambda g, p: (max(0, ari(g, p)) + nmi(g, p)) / 2

def quick_scores(Z, gt, T=None):
    Z = np.asarray(Z, np.float32); zr = np.abs(Z).sum(1) < 1e-9
    if zr.any(): Z[zr] = np.random.RandomState(0).normal(size=(int(zr.sum()), Z.shape[1])) * 1e-3
    Z = normalize(Z); L = linkage(Z, "average", "cosine")
    d, ix = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)
    out = {}
    for S in (40, 70):
        l = fcluster(L, S, "maxclust"); a = np.argsort(-d[:, -1])[: int(len(l) * 0.15)]; l[a] = 10**6 + np.arange(len(a)); out[f"agg{S}"] = sc(gt, l)
    m = gt >= 0; out["knn_pur"] = float(((gt[ix[:, 1:]] == gt[:, None]) & m[:, None]).mean(1)[m].mean())
    if T is not None:
        T = normalize(T); it = NearestNeighbors(n_neighbors=16, metric="cosine").fit(T).kneighbors(T)[1]
        out["rec15"] = float(np.mean([len(set(p[1:]) & set(q[1:])) / 15 for p, q in zip(ix, it)]))
    return out

def one(args):
    f, wp, fw, fc = args; s = load_subset(f, fw, fc, IDF); W = np.load(wp)
    r = quick_scores(s["X"] @ W, s["y"], s["T"]); r["arxiv"] = s["arxiv"]; return r

if __name__ == "__main__":
    wp, fw, fc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); rounds = [int(x) for x in sys.argv[4:]] or [71, 72, 73]
    with ProcessPoolExecutor(4) as ex: R = list(ex.map(one, [(f, wp, fw, fc) for f in files_for(rounds)]))
    for grp, sel in (("social", [r for r in R if not r["arxiv"]]), ("arxiv", [r for r in R if r["arxiv"]])):
        ks = [k for k in ("agg40", "agg70", "knn_pur", "rec15") if any(k in r for r in sel)]
        print(os.path.basename(wp), grp, " ".join("%s %.4f" % (k, np.mean([r[k] for r in sel if k in r])) for k in ks), flush=True)
