"""Error decomposition for the current pipeline (unseen rounds 69-73): value of better singleton/noise selection."""
import sys, numpy as np, pandas as pd, warnings
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
def one(f):
    s = load_subset(f, FW, FC, True); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); n = len(gt); res = {}; y = gt == -1; rng = np.random.default_rng(0)
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    k, S, fr = (35, 80, .4) if arx else (20, 120, .3)
    def st(Z):
        Zs = smooth(Z, k, .4, 4); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; return Zs, d, linkage(Zs, "average", "cosine")
    Zs, d, L = st(Z); out = np.argsort(-d)[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
    Zs, d, L = st(normalize(np.hstack([Z, .75 * P]))); l0 = fcluster(L, S, "maxclust"); o = np.argsort(-d)[: int(n * fr)]; m = len(o)
    def single(idx): l = l0.copy(); l[idx] = 10**6 + np.arange(len(idx)); return l
    res["current"] = sc(gt, single(o)); res["current precision"] = y[o].mean(); res["no singletons"] = sc(gt, l0)
    nz, cl = np.flatnonzero(y), np.flatnonzero(~y)
    for p in (.4, .5, .6, .7, .85, 1.0):
        a = min(int(m * p), len(nz)); idx = np.concatenate([rng.choice(nz, a, replace=False), rng.choice(cl, m - a, replace=False)])
        res[f"singletons at precision {p}"] = sc(gt, single(idx))
        l = l0.copy(); l[idx] = -1; res[f"shared -1 cluster at precision {p}"] = sc(gt, l)
    l = l0.copy(); l[y] = -1; res["oracle noise as -1 (all of it)"] = sc(gt, l)
    l = gt.copy(); l[o] = 10**6 + np.arange(m); res["oracle clusters + my singletons"] = sc(gt, l)
    l = gt.copy(); l[y] = l0[y] + 10**5; res["oracle clusters, noise by my clusters"] = sc(gt, l)
    return arx, res
if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    for k in R[0][1]:
        so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); print("%-40s round %.4f social %.4f arxiv %.4f" % (k, (3 * so + ar) / 4, so, ar))
