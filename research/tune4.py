import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
def fn(c):
    k, S_, fr = PAR[c["arx"]]; Z = c["Z"]; X = c["X"]; T = c["T"]; n = len(Z); out = {}; m = int(n * fr)
    Zs0 = smooth(Z, k); d0 = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs0).kneighbors(Zs0)[0][:, -1]; L0 = linkage(Zs0, "average", "cosine"); out0 = np.argsort(-d0)[:m]
    ps = fcluster(L0, 90, "maxclust"); ok = np.ones(n, bool); ok[out0] = False; ok &= np.bincount(ps)[ps] >= 10
    def fin(ZA, sel=None):
        Zs = smooth(ZA, k); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; l = fcluster(linkage(Zs, "average", "cosine"), S_, "maxclust")
        o = np.argsort(-d)[:m] if sel is None else sel(d, l); l2 = l.copy(); l2[o] = 10**6 + np.arange(len(o)); return l2
    def aug(P, w=.75): return normalize(np.hstack([Z, w * normalize(P - P.mean(1, keepdims=True))]))
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); base = aug(P); out["base"] = fin(base)
    # --- classifier alternatives
    out["cls LinearSVC C=0.3"] = fin(aug(LinearSVC(C=.3).fit(X[ok], ps[ok]).decision_function(X)))
    out["cls LinearSVC C=1"] = fin(aug(LinearSVC(C=1.0).fit(X[ok], ps[ok]).decision_function(X)))
    lr = LogisticRegression(C=20, max_iter=60).fit(X[ok], ps[ok]); out["cls logistic logits"] = fin(aug(lr.decision_function(X))); out["cls logistic sqrt-prob"] = fin(normalize(np.hstack([Z, .75 * normalize(np.sqrt(lr.predict_proba(X)))])))
    C = normalize(np.vstack([np.asarray(X[ok][ps[ok] == g].mean(0)) for g in np.unique(ps[ok])])); out["cls centroid cosine"] = fin(aug(X @ C.T, .75))
    # --- classifier features
    kw = dict(min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    Xw = normalize(TfidfVectorizer(ngram_range=(1, 2), max_features=60000, **kw).fit_transform(T)); out["feat words only"] = fin(aug(RidgeClassifier(alpha=10.0).fit(Xw[ok], ps[ok]).decision_function(Xw)))
    Xb = normalize(hstack([TfidfVectorizer(ngram_range=(1, 3), max_features=120000, **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), max_features=200000, **kw).fit_transform(T)]).tocsr())
    out["feat bigger (w1-3, c2-6, 320k)"] = fin(aug(RidgeClassifier(alpha=10.0).fit(Xb[ok], ps[ok]).decision_function(Xb)))
    Xs = normalize(hstack([X, 2.0 * Z]).tocsr()); out["feat tfidf + student embedding"] = fin(aug(RidgeClassifier(alpha=10.0).fit(Xs[ok], ps[ok]).decision_function(Xs)))
    # --- singleton selection relative to the cluster
    def rel(lam):
        def sel(d, l):
            g = np.argsort(np.argsort(d)) / n; ids, inv, cnt = np.unique(l, return_inverse=True, return_counts=True); w = np.zeros(n)
            for i in range(len(ids)):
                mk = inv == i; w[mk] = np.argsort(np.argsort(d[mk])) / max(cnt[i] - 1, 1)
            return np.argsort(-(lam * g + (1 - lam) * w))[:m]
        return sel
    for lam in (.85, .7, .5): out[f"singletons: {lam} global + {1-lam:.2f} within-cluster rank"] = fin(base, rel(lam))
    def sizeaware(p):
        def sel(d, l):
            cnt = np.bincount(l)[l]; return np.argsort(-(np.argsort(np.argsort(d)) / n - p * np.argsort(np.argsort(cnt)) / n))[:m]
        return sel
    for p in (.15, .3, -.15): out[f"singletons: density rank - {p} * cluster-size rank"] = fin(base, sizeaware(p))
    return out
if __name__ == "__main__":
    r = evaluate(fn, sets=("A", "B", "C"), jobs=10); b = r["base"]; print()
    for k, v in r.items():
        if k != "base": print("  %-58s" % k, " ".join("%+.4f" % (v[i][0] - b[i][0]) for i in range(3)), "  <== consistent" if all(v[i][0] > b[i][0] for i in range(3)) else "")
