import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
def fn(c):
    k, S_, fr = PAR[c["arx"]]; ZA = c["ZA"]; n = len(ZA); out = {}
    Zs = smooth(ZA, k); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; L = linkage(Zs, "average", "cosine"); l0 = fcluster(L, S_, "maxclust"); o = np.argsort(-d)[: int(n * fr)]
    l = l0.copy(); l[o] = 10**6 + np.arange(len(o)); out["base"] = l
    Lo = linkage(Zs[o], "average", "cosine")
    for div in (1.5, 2, 3, 5):
        l = l0.copy(); l[o] = 10**6 + fcluster(Lo, max(2, int(len(o) / div)), "maxclust"); out[f"outliers in groups of ~{div}"] = l
    for t in (.15, .25, .35):
        l = l0.copy(); l[o] = 10**6 + fcluster(Lo, t, "distance"); out[f"outliers merged below distance {t}"] = l
    # fine tree cut for outliers: keep their membership in a much finer cut of the full tree
    for fine in (600, 1200, 2000):
        lf = fcluster(L, fine, "maxclust"); l = l0.copy(); l[o] = 10**6 + lf[o]; out[f"outliers by fine cut {fine}"] = l
    return out
if __name__ == "__main__":
    r = evaluate(fn, sets=("A", "B", "C"), jobs=10); b = r["base"]
    for k, v in r.items():
        if k != "base": print("  %-36s" % k, " ".join("%+.4f" % (v[i][0] - b[i][0]) for i in range(3)))
