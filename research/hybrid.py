"""Hybrid: my clusters + a shared noise cluster voted by several UMAP+HDBSCAN runs on the same embedding."""
import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
import umap, hdbscan
def fn(c):
    k, S_, fr = PAR[c["arx"]]; ZA = c["ZA"]; n = len(ZA); gt = c["gt"]; y = gt == -1; out = {}
    base = baseline(c); out["baseline"] = base; single = base >= 10**6
    votes = np.zeros(n); labs = []
    for seed in range(4):
        U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed, n_jobs=1).fit_transform(ZA); l = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=8).fit_predict(U); votes += l == -1; labs.append(l)
    out["umap+hdbscan (seed 0) noise=-1"] = labs[0]
    for v in (1, 2, 3, 4):
        m = votes >= v; out[f"P votes>={v}: precision"] = ("stat", float(y[m].mean()) if m.any() else 0.0); out[f"P votes>={v}: share of points"] = ("stat", float(m.mean()))
        l = base.copy(); l[m] = -1; out[f"my clusters, votes>={v} -> shared -1"] = l
        l = base.copy(); l[m & single] = -1; out[f"my singletons with votes>={v} -> shared -1"] = l
        l = labs[0].copy(); mm = l == -1; l[mm] = 10**6 + np.arange(mm.sum()); l[m] = -1; out[f"umap+hdbscan clusters, votes>={v} -> -1, other noise singletons"] = l
    return out
def _run2(c):
    o = fn(c); return c["arx"], {k: (v[1] if isinstance(v, tuple) else sc(c["gt"], v)) for k, v in o.items()}
if __name__ == "__main__":
    for name in ("A", "B"):
        D = load(name)
        with ProcessPoolExecutor(8) as ex: R = list(ex.map(_run2, D))
        print("== set", name)
        for kk in R[0][1]:
            so = np.mean([o[kk] for a, o in R if not a]); ar = np.mean([o[kk] for a, o in R if a]); print("%-62s %.4f (soc %.4f arx %.4f)" % (kk, (3 * so + ar) / 4, so, ar), flush=True)
