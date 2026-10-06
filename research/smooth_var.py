"""Variants of the neighbour-smoothing step (table fixed, self-training on)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from feats import prep
from grid2 import fixz, sc
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from scipy.sparse import hstack
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
WP, FW, FC = sys.argv[1], 8192, 16384
def knn(Z, k):
    S = Z @ Z.T; np.fill_diagonal(S, -2); ix = np.argpartition(-S, k, axis=1)[:, :k]; sv = np.take_along_axis(S, ix, 1); o = np.argsort(-sv, 1)
    return np.take_along_axis(ix, o, 1), np.take_along_axis(sv, o, 1)
def smooth(Z, k, a, it, mode):
    ix, sv = knn(Z, k)
    if mode == "snn":   # re-rank 3k candidates by shared-neighbour count
        ix3, _ = knn(Z, 3 * k); n = len(Z); A = np.zeros((n, n), np.float32); np.put_along_axis(A, ix, 1, 1)
        shared = np.einsum("nkd,nd->nk", A[ix3], A, optimize=True) if n <= 6000 else None
        o = np.argsort(-shared, 1)[:, :k]; ix = np.take_along_axis(ix3, o, 1)
    for i in range(it):
        if mode == "dyn" and i: ix, sv = knn(Z, k)
        if mode == "wt": w = np.maximum(sv, 0) ** 2; nb = (Z[ix] * w[:, :, None]).sum(1) / (w.sum(1, keepdims=True) + 1e-9)
        elif mode == "mutual":
            n = len(Z); A = np.zeros((n, n), bool); np.put_along_axis(A, ix, True, 1); M = (A & A.T)[np.arange(n)[:, None], ix].astype(np.float32) + 0.15
            nb = (Z[ix] * M[:, :, None]).sum(1) / M.sum(1, keepdims=True)
        else: nb = Z[ix].mean(1)
        Z = normalize((1 - a) * Z + a * nb)
    return Z
def one(f):
    s = load_subset(f, FW, FC, True); gt = s["y"]; arx = s["arxiv"]; Z0 = fixz(s["X"] @ np.load(WP)); n = len(gt); res = {}
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    k, S, fr = (35, 80, .4) if arx else (20, 120, .3)
    for mode, kk, a, it in (("mean", k, .4, 4), ("wt", k, .4, 4), ("mutual", k, .4, 4), ("dyn", k, .4, 4), ("snn", k, .4, 4), ("mean", k, .3, 6), ("wt", 2 * k, .4, 4), ("dyn", k, .3, 6), ("mutual", 2 * k, .4, 4)):
        def st(Z):
            Zs = smooth(Z, kk, a, it, mode); d = 1 - knn(Zs, 15)[1][:, -1]; return d, linkage(Zs, "average", "cosine")
        d, L = st(Z0); out = np.argsort(-d)[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
        P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
        d, L = st(normalize(np.hstack([Z0, .75 * P]))); l = fcluster(L, S, "maxclust"); o = np.argsort(-d)[: int(n * fr)]; l[o] = 10**6 + np.arange(len(o)); res[(mode, kk // k, a, it)] = sc(gt, l)
    return arx, res
if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    for k in R[0][1]:
        so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); print("%-26s round %.4f social %.4f arxiv %.4f" % (k, (3 * so + ar) / 4, so, ar))
