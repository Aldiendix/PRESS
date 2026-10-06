import sys, numpy as np, warnings, itertools
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
sc = lambda g, p: (max(0, ari(g, p)) + nmi(g, p)) / 2
def smooth(Z, k, a, it):
    if not it: return Z
    ix = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z).kneighbors(Z)[1][:, 1:]
    for _ in range(it): Z = normalize((1 - a) * Z + a * Z[ix].mean(1))
    return Z
def one(f):
    s = load_subset(f, 8192, 4096); Z0 = np.asarray(s["X"] @ np.load(sys.argv[1]), np.float32); zr = np.abs(Z0).sum(1) < 1e-9; Z0[zr] = np.random.RandomState(0).normal(size=(int(zr.sum()), Z0.shape[1])) * 1e-3; Z0 = normalize(Z0); gt = s["y"]; out = {}
    for (k, a, it) in ((0, 0, 0), (12, .2, 2), (20, .4, 4), (35, .4, 4), (20, .6, 6)):
        Z = smooth(Z0, k, a, it); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)[0][:, -1]
        for meth in ("average", "ward"):
            L = linkage(Z, meth, "cosine" if meth == "average" else "euclidean")
            for S in (30, 50, 80, 120):
                l0 = fcluster(L, S, "maxclust")
                for fr in (0, .1, .2, .3):
                    l = l0.copy(); a_ = np.argsort(-d)[: int(len(l) * fr)]; l[a_] = 10**6 + np.arange(len(a_)); out[(k, a, it, meth, S, fr)] = sc(gt, l)
    return s["arxiv"], out
if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    for grp in (False, True):
        sel = [o for a, o in R if a == grp]; keys = sel[0].keys()
        m = sorted(((np.mean([o[k] for o in sel]), k) for k in keys), reverse=True)
        print("ARXIV" if grp else "SOCIAL")
        for v, k in m[:8]: print("  %.4f" % v, k)
        print("  ... no-smooth avg S50 fr.1: %.4f" % np.mean([o[(0, 0, 0, 'average', 50, .1)] for o in sel]))
