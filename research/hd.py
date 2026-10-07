import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
import hdbscan
def fn(c):
    k, S_, fr = PAR[c["arx"]]; ZA = c["ZA"]; n = len(ZA); out = {"base": baseline(c)}; gt = c["gt"]
    for it in (4, 10, 20):
        Zs = smooth(ZA, k, .4, it)
        for ms in (5, 15):
            for meth in ("eom", "leaf"):
                l = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=ms, cluster_selection_method=meth).fit_predict(Zs.astype(np.float64)); nm = f"smooth x{it} + HDBSCAN ms{ms} {meth}"
                out[nm + " | noise=-1"] = l; l2 = l.copy(); m = l2 == -1; l2[m] = 10**6 + np.arange(m.sum()); out[nm + " | noise=singletons"] = l2
                # my clusters, HDBSCAN noise as shared cluster
                l3 = out["base"].copy(); l3[m] = -1; out[nm + " | my clusters + its noise as -1"] = l3
    return out
if __name__ == "__main__":
    r = evaluate(fn, sets=("A", "B"), jobs=10); b = r["base"]; print()
    for k_, v in sorted(r.items(), key=lambda kv: -kv[1][0][0])[:12]: print("  %-62s" % k_, " ".join("%.4f" % v[i][0] for i in range(2)))
