"""Can a high-precision subset of noise be found and grouped into one shared cluster?"""
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
    s = load_subset(f, FW, FC, True); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); n = len(gt); y = gt == -1
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    k, S, fr = (35, 80, .4) if arx else (20, 120, .3)
    def st(Z):
        Zs = smooth(Z, k, .4, 4); dd, ix = NearestNeighbors(n_neighbors=31, metric="cosine").fit(Zs).kneighbors(Zs); return Zs, dd, ix, linkage(Zs, "average", "cosine")
    Zs, dd, ix, L = st(Z); out = np.argsort(-dd[:, 15])[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); Pn = normalize(P - P.mean(1, keepdims=True))
    Zs, dd, ix, L = st(normalize(np.hstack([Z, .75 * Pn]))); d = dd[:, 15]; l0 = fcluster(L, S, "maxclust"); order = np.argsort(-d); m = int(n * fr)
    base = l0.copy(); base[order[:m]] = 10**6 + np.arange(m)
    srt = np.sort(P, 1); margin = srt[:, -1] - srt[:, -2]; pur = (l0[ix[:, 1:]] == l0[:, None]).mean(1); d_raw = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)[0][:, -1]
    rk = lambda v: v.argsort().argsort() / n
    scores = dict(knn=d, knn_raw=d_raw, margin=-margin, maxp=-srt[:, -1], purity=-pur, combo=rk(d) + rk(-margin) + rk(d_raw), combo2=rk(d) + rk(-srt[:, -1]))
    res = {"base": sc(gt, base), "base_rate": y.mean()}
    for nm, v in scores.items():
        o = np.argsort(-v)
        for q in (.02, .05, .1, .2):
            top = o[: int(n * q)]; res[("prec", nm, q)] = y[top].mean()
            l = base.copy(); l[top] = -1; res[("score", nm, q)] = sc(gt, l)
    return arx, res
if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    avg = lambda k: (3 * np.mean([o[k] for a, o in R if not a]) + np.mean([o[k] for a, o in R if a])) / 4
    print("base %.4f  noise base rate %.3f" % (avg("base"), avg("base_rate")))
    for nm in ("knn", "knn_raw", "margin", "maxp", "purity", "combo", "combo2"):
        print("%-8s" % nm, " ".join("q%.2f: prec %.2f score %.4f |" % (q, avg(("prec", nm, q)), avg(("score", nm, q))) for q in (.02, .05, .1, .2)))
