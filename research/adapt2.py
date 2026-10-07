"""(1) per-batch adaptive settings from batch statistics; (2) oracle headroom of non-uniform tree cuts. Rounds 61-74 unseen."""
import sys, numpy as np, pandas as pd, pickle; sys.path.insert(0, "research")
import search as S
from lab import sc, smooth
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeClassifier
from scipy.cluster.hierarchy import fcluster, linkage
SS = (60, 90, 120, 160, 220); FR = (.1, .15, .2, .25, .3, .35, .4)
def job(i):
    c = S.D[i]; p = dict(S.V4[c["arx"]]); Z, X, gt, n = c["Z"], c["X"], c["gt"], len(c["gt"])
    Zs = smooth(Z, p["k"]); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; L = linkage(Zs, "average", "cosine")
    out = np.argsort(-d)[: int(n * p["fr1"])]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); ZA = normalize(np.hstack([Z, .75 * normalize(P - P.mean(1, keepdims=True))]))
    Zs = smooth(ZA, p["k"]); dd = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0]; d = dd[:, -1]; L = linkage(Zs, "average", "cosine"); order = np.argsort(-d); grid = {}
    for s_ in SS:
        l0 = fcluster(L, s_, "maxclust")
        for fr in FR:
            l = l0.copy(); o = order[: int(n * fr)]; l[o] = 10**6 + np.arange(len(o)); grid[(s_, fr)] = sc(gt, l)
    # oracle merge of a fine cut: every fine cluster goes to its majority ground-truth label (noise-majority pieces stay separate)
    fr = p["fr"] if not c["arx"] else .3; o = order[: int(n * fr)]; keep = np.ones(n, bool); keep[o] = False; orc = {}
    for fine in (300, 600):
        lf = fcluster(L, fine, "maxclust"); l = np.empty(n, np.int64)
        for g in np.unique(lf):
            m = lf == g; mk = m & keep
            maj = np.bincount(gt[mk] + 1).argmax() - 1 if mk.any() else -1
            l[m] = maj if maj >= 0 else 10**5 + g
        l[o] = 10**6 + np.arange(len(o)); orc[fine] = sc(gt, l)
    T = pd.read_parquet(c["f"], columns=["text"]).text; ln = T.str.len().values
    rs = np.random.default_rng(0).choice(n, 1500, replace=False); rel = d.mean() / (1 - Zs[rs] @ Zs[rs].T)[np.triu_indices(1500, 1)].mean()
    st = dict(medlen=np.median(ln), p10len=np.percentile(ln, 10), p90len=np.percentile(ln, 90), newline=T.str.contains("\n").mean(), url=T.str.contains("http").mean(), mention=T.str.contains("@").mean(),
              rel=rel, dmean=d.mean(), dstd=d.std() / d.mean(), d1=dd[:, 1].mean(), big=np.sort(np.bincount(fcluster(L, 40, "maxclust")))[-1] / n, noise=(gt == -1).mean(), k=gt.max() + 1)
    return c["arx"], c["rnd"], grid, orc, st
if __name__ == "__main__":
    S.D = S.load()
    with ProcessPoolExecutor(14) as ex: R = list(ex.map(job, range(len(S.D))))
    pickle.dump(R, open("cache/adapt2.pkl", "wb"))
    for arx in (False, True):
        sel = [r for r in R if r[0] == arx]; keys = list(sel[0][2]); M = np.array([[r[2][k] for k in keys] for r in sel]); b = M.mean(0).argmax(); cur = keys.index((120, .2) if not arx else (90, .3))
        print("ARXIV" if arx else "SOCIAL", len(sel), "subsets | current %.4f | best fixed %s %.4f | per-subset oracle %.4f | oracle merge of fine cut 300: %.4f, 600: %.4f" % (M[:, cur].mean(), keys[b], M[:, b].mean(), M.max(1).mean(), np.mean([r[3][300] for r in sel]), np.mean([r[3][600] for r in sel])))
        bestfr = np.array([keys[i][1] for i in M.argmax(1)]); bestS = np.array([keys[i][0] for i in M.argmax(1)])
        print("   correlation of batch statistics with the per-subset best singleton share / cluster count:")
        for k in sel[0][4]: v = np.array([r[4][k] for r in sel]); print("     %-8s fr %+.2f  S %+.2f" % (k, np.corrcoef(v, bestfr)[0, 1], np.corrcoef(v, bestS)[0, 1]))
