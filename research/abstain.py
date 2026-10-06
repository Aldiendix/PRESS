"""Which points should become singletons? Oracle headroom and candidate ranking signals (unseen rounds 69-73)."""
import sys, numpy as np, pandas as pd, warnings, pickle
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from feats import prep
from grid2 import smooth, fixz, sc
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from scipy.sparse import hstack
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
WP, FW, FC = sys.argv[1], 8192, 16384
ROUNDS = [int(x) for x in sys.argv[2:]] or [69, 70, 71, 72, 73]
def one(f):
    s = load_subset(f, FW, FC, True); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); n = len(gt); res = {}; y = gt == -1
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    k, S, fr = (35, 80, .4) if arx else (20, 120, .3)
    def st(Z):
        Zs = smooth(Z, k, .4, 4); dd, ix = NearestNeighbors(n_neighbors=31, metric="cosine").fit(Zs).kneighbors(Zs); return Zs, dd, ix, linkage(Zs, "average", "cosine")
    Zs0, dd0, ix0, L = st(Z); d0 = dd0[:, 15]; out = np.argsort(-d0)[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); Pn = normalize(P - P.mean(1, keepdims=True))
    Zs, dd, ix, L = st(normalize(np.hstack([Z, .75 * Pn]))); d = dd[:, 15]; l0 = fcluster(L, S, "maxclust"); m = int(n * fr)
    def single(idx): l = l0.copy(); l[idx] = 10**6 + np.arange(len(idx)); return sc(gt, l)
    ids, inv, cnt = np.unique(l0, return_inverse=True, return_counts=True)
    maj = np.array([np.bincount(gt[l0 == i] + 1).argmax() - 1 for i in ids])[inv]; bad = y | (maj != gt)
    C = normalize(np.stack([Zs[l0 == i].mean(0) for i in ids])); cs = (Zs * C[inv]).sum(1); srt = np.sort(P, 1)
    d_raw = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)[0][:, -1]
    feats = dict(knn=d, knn5=dd[:, 5], knn30=dd[:, 30], knn_first=d0, knn_raw=d_raw, margin=-(srt[:, -1] - srt[:, -2]), maxp=-srt[:, -1], centroid=-cs, small=-np.log(cnt[inv]),
                 purity=-(l0[ix[:, 1:16]] == l0[:, None]).mean(1), length=-np.log1p([len(t) for t in T]), znorm=-np.linalg.norm(s["X"] @ np.load(WP), axis=1))
    res["current"] = single(np.argsort(-d)[:m]); res["bad rate"] = bad.mean(); res["current precision for bad"] = bad[np.argsort(-d)[:m]].mean()
    b = np.flatnonzero(bad); g = np.flatnonzero(~bad); rng = np.random.default_rng(0)
    res["oracle: all bad points, exactly"] = single(b)
    for p in (.5, .6, .7, .8):
        a = min(int(m * p), len(b)); res[f"m points at precision {p} for bad"] = single(np.concatenate([rng.choice(b, a, replace=False), rng.choice(g, m - a, replace=False)]))
    for nm, v in feats.items(): o = np.argsort(-v)[:m]; res[("sig", nm)] = (single(o), bad[o].mean())
    return arx, res, np.column_stack([feats[k] for k in feats]), bad, y, list(feats), l0, gt
if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, files_for(ROUNDS)))
    pickle.dump(R, open("cache/abstain_%d_%d.pkl" % (ROUNDS[0], ROUNDS[-1]), "wb"))
    for k in R[0][1]:
        f = (lambda o: o[k][0]) if isinstance(k, tuple) else (lambda o: o[k])
        so = np.mean([f(o) for a, o, *_ in R if not a]); ar = np.mean([f(o) for a, o, *_ in R if a])
        extra = "  precision %.2f" % np.mean([o[k][1] for a, o, *_ in R]) if isinstance(k, tuple) else ""
        print("%-36s round %.4f social %.4f arxiv %.4f%s" % (k if not isinstance(k, tuple) else "signal " + k[1], (3 * so + ar) / 4, so, ar, extra))
