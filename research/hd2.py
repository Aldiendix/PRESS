import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
import hdbscan
def fn(c):
    k, S_, fr = PAR[c["arx"]]; ZA = c["ZA"]; n = len(ZA); out = {}; m = int(n * fr)
    Zs = smooth(ZA, k); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; la = fcluster(linkage(Zs, "average", "cosine"), S_, "maxclust"); o = np.argsort(-d)[:m]; dens_out = np.zeros(n, bool); dens_out[o] = True
    def mk(l, mask): l = np.asarray(l).copy(); idx = np.flatnonzero(mask); l[idx] = 10**6 + np.arange(len(idx)); return l
    out["base"] = mk(la, dens_out)
    for ms in (8, 15, 30):
        h = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=ms).fit(Zs.astype(np.float64)); lh = h.labels_; hn = lh == -1; t = f"ms{ms}"
        out[f"{t} H clusters, H noise singletons"] = mk(lh, hn)
        out[f"{t} H clusters, singletons = H noise OR density"] = mk(lh, hn | dens_out)
        out[f"{t} H clusters, singletons = density only (H noise -> avg-link cluster)"] = mk(np.where(hn, la + 10**4, lh), dens_out)
        out[f"{t} avg-link clusters, singletons = H noise OR density"] = mk(la, hn | dens_out)
        out[f"{t} avg-link clusters, singletons = H noise AND density, rest by density to share"] = mk(la, hn & dens_out)
        prod = lh.astype(np.int64) * 1000 + la; out[f"{t} product partition, singletons = density"] = mk(prod, dens_out)
        out[f"{t} product partition, singletons = H noise OR density"] = mk(prod, hn | dens_out)
        # H clusters split by avg-link only when the avg-link piece is big
        cnt = np.unique(prod, return_counts=True); big = dict(zip(*cnt)); piece = np.array([big[p] for p in prod])
        out[f"{t} H clusters, split off avg-link pieces >= 25, singletons = density"] = mk(np.where(piece >= 25, prod, lh.astype(np.int64) * 1000), dens_out | (hn & (piece < 25)))
        pr = h.probabilities_; lowp = (pr < np.quantile(pr[~hn], .15)) & ~hn if (~hn).any() else hn
        out[f"{t} H clusters, singletons = H noise + its 15% weakest members"] = mk(lh, hn | lowp)
    return out
if __name__ == "__main__":
    r = evaluate(fn, sets=("A", "B", "C"), jobs=10); b = r["base"]; print()
    for k_, v in sorted(r.items(), key=lambda kv: -sum(x[0] for x in kv[1]))[:14]: print("  %-76s" % k_, " ".join("%.4f" % v[i][0] for i in range(3)))
