import sys, numpy as np, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
import umap, hdbscan
from train import load_subset, files_for
from grid2 import smooth, fixz, sc
from concurrent.futures import ProcessPoolExecutor
def one(f):
    s = load_subset(f, 8192, 4096); gt = s["y"]; Z0 = fixz(s["X"] @ np.load("cache/W_a.npy")); out = {}
    for sm in (0, 1):
        Z = smooth(Z0, 20, .4, 4) if sm else Z0
        for nn in (15, 30):
            U = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric="cosine", random_state=1, n_jobs=1).fit_transform(Z)
            for mcs, ms in ((25, 5), (25, 10), (25, 25), (40, 10)):
                l = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(U); out[(sm, nn, mcs, ms, "noise")] = sc(gt, l)
                l2 = l.copy(); n = l2 == -1; l2[n] = 10**6 + np.arange(n.sum()); out[(sm, nn, mcs, ms, "single")] = sc(gt, l2)
    return s["arxiv"], out
if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex: R = list(ex.map(one, files_for([71, 72, 73])))
    rows = sorted(((3 * np.mean([o[k] for a, o in R if not a]) + np.mean([o[k] for a, o in R if a])) / 4, np.mean([o[k] for a, o in R if not a]), np.mean([o[k] for a, o in R if a]), k) for k in R[0][1])
    for r in rows[::-1][:8]: print("round %.4f social %.4f arxiv %.4f" % r[:3], r[3])
