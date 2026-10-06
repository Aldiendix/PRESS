"""More test-time (transductive) variants on a fixed table; unseen rounds 69-73."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from feats import prep
from grid2 import smooth, fixz, sc
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, Ridge
from scipy.sparse import hstack
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
WP, FW, FC = sys.argv[1], 8192, 16384
def one(f):
    s = load_subset(f, FW, FC, True); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); n = len(gt); res = {}
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    k, S, fr = (35, 80, .4) if arx else (20, 120, .3)
    def st(Z):
        Zs = smooth(Z, k, .4, 4); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; return Zs, d, linkage(Zs, "average", "cosine")
    def fin(Zc):
        Zs, d, L = st(Zc); l = fcluster(L, S, "maxclust"); o = np.argsort(-d)[: int(n * fr)]; l2 = l.copy(); l2[o] = 10**6 + np.arange(len(o)); return sc(gt, l2), l, o, Zs
    Zs, d, L = st(Z); out = np.argsort(-d)[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
    ZA = normalize(np.hstack([Z, .75 * P])); res["A self-train"], lA, oA, ZsA = fin(ZA)
    for al in (1.0, 3.0):
        R = fixz(Ridge(alpha=al).fit(X, Zs).predict(X))
        for w in (.5, 1.0):
            res[f"B ridge->Zs a{al} w{w}"] = fin(normalize(np.hstack([Z, w * R])))[0]
        res[f"C A+B a{al}"] = fin(normalize(np.hstack([Z, .75 * P, .5 * R])))[0]
        R2 = fixz(Ridge(alpha=al).fit(X, ZsA).predict(X)); res[f"E A then ridge->ZsA a{al}"] = fin(normalize(np.hstack([ZA, .5 * R2])))[0]
    # D: reassign dense points with a classifier trained on the final clusters
    keep = np.ones(n, bool); keep[oA] = False; big = keep & (np.bincount(lA)[lA] >= 5)
    pred = RidgeClassifier(alpha=10.0).fit(X[big], lA[big]).predict(X); l = lA.copy(); l[keep] = pred[keep]; l[oA] = 10**6 + np.arange(len(oA)); res["D reassign by classifier"] = sc(gt, l)
    # F: second self-training round with more classes on the refined space
    ps2 = fcluster(linkage(ZsA, "average", "cosine"), 150, "maxclust"); ok2 = keep & (np.bincount(ps2)[ps2] >= 8)
    P2 = RidgeClassifier(alpha=10.0).fit(X[ok2], ps2[ok2]).decision_function(X); P2 = normalize(P2 - P2.mean(1, keepdims=True))
    res["F 2nd round 150 classes"] = fin(normalize(np.hstack([Z, .5 * P, .6 * P2])))[0]
    return arx, res
if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    for k in R[0][1]:
        so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); print("%-30s round %.4f social %.4f arxiv %.4f" % (k, (3 * so + ar) / 4, so, ar))
