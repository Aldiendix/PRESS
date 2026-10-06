"""Alternative final clusterers on the self-trained embedding (table fixed; unseen rounds 69-73)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
import umap, hdbscan
from train import load_subset, files_for
from feats import prep
from grid2 import smooth, fixz, sc
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans
from scipy.sparse import hstack
from scipy.sparse.linalg import eigsh
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
        Zs = smooth(Z, k, .4, 4); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; return Zs, d
    Zs, d = st(Z); L = linkage(Zs, "average", "cosine"); out = np.argsort(-d)[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
    ZA = normalize(np.hstack([Z, .75 * P])); Zs, d = st(ZA); o = np.argsort(-d)[: int(n * fr)]
    def fin(l): l = np.asarray(l).copy(); l[o] = 10**6 + np.arange(len(o)); return sc(gt, l)
    res["average (current)"] = fin(fcluster(linkage(Zs, "average", "cosine"), S, "maxclust"))
    res["ward"] = fin(fcluster(linkage(Zs, "ward"), S, "maxclust")); res["complete"] = fin(fcluster(linkage(Zs, "complete", "cosine"), S, "maxclust"))
    res["weighted"] = fin(fcluster(linkage(Zs, "weighted", "cosine"), S, "maxclust"))
    for kk in (40, S): res[f"kmeans {'S' if kk == S else 40}"] = fin(KMeans(kk, n_init=3, random_state=0).fit_predict(Zs))
    G = kneighbors_graph(Zs, 15, metric="cosine"); G = ((G + G.T) > 0).astype(np.float32); dg = np.asarray(G.sum(1)).ravel(); Dm = 1 / np.sqrt(dg); A = G.multiply(Dm[:, None]).multiply(Dm[None, :]).tocsr()
    w, v = eigsh(A, k=40, which="LA", tol=1e-3); V = normalize(v); res["spectral40+kmeans60"] = fin(KMeans(60, n_init=3, random_state=0).fit_predict(V)); res["spectral40+average"] = fin(fcluster(linkage(V, "average", "cosine"), S, "maxclust"))
    for src, nm in ((ZA, "raw"), (Zs, "smoothed")):
        U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=1, n_jobs=1).fit_transform(src)
        for ms in (5, 10):
            l = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=ms).fit_predict(U); res[f"umap+hdbscan {nm} ms{ms} noise=-1"] = sc(gt, l)
            l2 = l.copy(); m = l2 == -1; l2[m] = 10**6 + np.arange(m.sum()); res[f"umap+hdbscan {nm} ms{ms} noise=singletons"] = sc(gt, l2)
        res[f"umap5d {nm} + average"] = fin(fcluster(linkage(U, "average"), S, "maxclust"))
    return arx, res
if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    for k in R[0][1]:
        so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); print("%-42s round %.4f social %.4f arxiv %.4f" % (k, (3 * so + ar) / 4, so, ar))
