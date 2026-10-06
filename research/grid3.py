"""Clustering-stage variants on a fixed table."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from grid2 import smooth, fixz, sc
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
WP = sys.argv[1]
def single(l, idx): l = l.copy(); l[idx] = 10**6 + np.arange(len(idx)); return l
def one(f):
    s = load_subset(f, 8192, 4096); gt = s["y"]; arx = s["arxiv"]; Z0 = fixz(s["X"] @ np.load(WP)); out = {}; n = len(gt)
    d0 = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z0).kneighbors(Z0)[0][:, -1]
    for (k, a, it) in ((20, .4, 4), (35, .4, 4), (20, .5, 8), (50, .5, 6)):
        Z = smooth(Z0, k, a, it); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)[0][:, -1]
        L = linkage(Z, "average", "cosine")
        for S in (50, 80, 120, 200):
            l0 = fcluster(L, S, "maxclust"); ids, inv, cnt = np.unique(l0, return_inverse=True, return_counts=True)
            C = normalize(np.stack([Z[l0 == i].mean(0) for i in ids])); cs = (Z * C[inv]).sum(1)
            for fr in (.1, .2, .3, .4):
                m = int(n * fr); key = (k, a, it, S, fr)
                out[key + ("knn",)] = sc(gt, single(l0, np.argsort(-d)[:m]))
                out[key + ("knn_raw",)] = sc(gt, single(l0, np.argsort(-d0)[:m]))
                out[key + ("centroid",)] = sc(gt, single(l0, np.argsort(cs)[:m]))
                small = cnt[inv] < 15; l1 = single(l0, np.flatnonzero(small)); rest = np.argsort(-(d - 10 * small))[: max(0, m - small.sum())]
                out[key + ("small15+knn",)] = sc(gt, single(l1, rest))
                # merge small clusters into nearest big centroid, then knn singletons
                big = cnt >= 15
                if big.sum() >= 2:
                    l2 = l0.copy(); sm = np.flatnonzero(small); l2[sm] = ids[big][(Z[sm] @ C[big].T).argmax(1)]
                    out[key + ("merge15+knn",)] = sc(gt, single(l2, np.argsort(-d)[:m]))
    return arx, out
if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex: R = list(ex.map(one, files_for([66, 67, 68, 69, 70, 71, 72, 73])))
    keys = set.intersection(*[set(o) for _, o in R])
    for grp in (False, True):
        sel = [o for a, o in R if a == grp]; m = sorted(((np.mean([o[k] for o in sel]), k) for k in keys), reverse=True)
        print("ARXIV" if grp else "SOCIAL")
        for v, k in m[:10]: print("  %.4f" % v, k)
        for meth in ("knn", "knn_raw", "centroid", "small15+knn", "merge15+knn"):
            b = max((v, k) for v, k in m if k[-1] == meth); print("   best %-12s %.4f" % (meth, b[0]), b[1][:5])
