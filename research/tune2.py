"""Fine tuning of self-training and final-stage settings on three unseen sets (A, B, C). Accept only consistent wins."""
import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
from sklearn.linear_model import RidgeClassifier
def fn(c):
    k, S_, fr = PAR[c["arx"]]; Z = c["Z"]; X = c["X"]; n = len(Z); out = {}
    Zs0 = smooth(Z, k); dd0 = NearestNeighbors(n_neighbors=31, metric="cosine").fit(Zs0).kneighbors(Zs0)[0]; L0 = linkage(Zs0, "average", "cosine"); out0 = np.argsort(-dd0[:, 15])[: int(n * fr)]
    def P_of(Sp, al, minsz=10, excl=True):
        ps = fcluster(L0, Sp, "maxclust"); ok = np.ones(n, bool)
        if excl: ok[out0] = False
        ok &= np.bincount(ps)[ps] >= minsz; P = RidgeClassifier(alpha=al).fit(X[ok], ps[ok]).decision_function(X); return normalize(P - P.mean(1, keepdims=True))
    def fin(ZA, S=S_, f=fr, kn=15, kk=k):
        Zs = smooth(ZA, kk); d = NearestNeighbors(n_neighbors=31, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, kn]; l = fcluster(linkage(Zs, "average", "cosine"), S, "maxclust"); o = np.argsort(-d)[: int(n * f)]; l[o] = 10**6 + np.arange(len(o)); return l
    P90 = P_of(90, 10.0); base = normalize(np.hstack([Z, .75 * P90])); out["base"] = fin(base)
    for Sp in (60, 150, 250): out[f"ST classes {Sp}"] = fin(normalize(np.hstack([Z, .75 * P_of(Sp, 10.0)])))
    for al in (3.0, 30.0): out[f"ST alpha {al}"] = fin(normalize(np.hstack([Z, .75 * P_of(90, al)])))
    for w in (.5, 1.0, 1.3): out[f"ST weight {w}"] = fin(normalize(np.hstack([Z, w * P90])))
    out["ST train on all points"] = fin(normalize(np.hstack([Z, .75 * P_of(90, 10.0, excl=False)])))
    out["ST min cluster 5"] = fin(normalize(np.hstack([Z, .75 * P_of(90, 10.0, minsz=5)])))
    out["ST two granularities 60+150"] = fin(normalize(np.hstack([Z, .55 * P_of(60, 10.0), .55 * P_of(150, 10.0)])))
    out["ST three 40+90+200"] = fin(normalize(np.hstack([Z, .45 * P_of(40, 10.0), .45 * P90, .45 * P_of(200, 10.0)])))
    for kn in (8, 25): out[f"density k {kn}"] = fin(base, kn=kn)
    for m in (.75, 1.25): out[f"singleton share x{m}"] = fin(base, f=fr * m)
    for m in (.7, 1.4): out[f"clusters x{m}"] = fin(base, S=int(S_ * m))
    return out
if __name__ == "__main__":
    r = evaluate(fn, sets=("A", "B", "C"), jobs=10); b = r["base"]
    print("\nconsistent wins (all three sets >= base):")
    for k, v in r.items():
        if k != "base" and all(v[i][0] >= b[i][0] for i in range(3)): print("  %-30s" % k, " ".join("%+.4f" % (v[i][0] - b[i][0]) for i in range(3)))
