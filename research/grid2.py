"""View-combination experiments on top of a student table (fixed clusterer)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from feats import prep
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from scipy.cluster.hierarchy import fcluster, linkage
from concurrent.futures import ProcessPoolExecutor
sc = lambda g, p: (max(0, ari(g, p)) + nmi(g, p)) / 2
WP, FW, FC = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else ("cache/W_a.npy", 8192, 4096)
def smooth(Z, k, a, it):
    ix = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z).kneighbors(Z)[1][:, 1:]
    for _ in range(it): Z = normalize((1 - a) * Z + a * Z[ix].mean(1))
    return Z
def fixz(Z):
    Z = np.asarray(Z, np.float32); zr = np.abs(Z).sum(1) < 1e-9; Z[zr] = np.random.RandomState(0).normal(size=(int(zr.sum()), Z.shape[1])) * 1e-3; return normalize(Z)
def clus(Z, gt, arx):
    k, S, fr = (35, 80, .3) if arx else (20, 80, .2)
    Z = smooth(Z, k, .4, 4); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)[0][:, -1]
    l = fcluster(linkage(Z, "average", "cosine"), S, "maxclust"); a_ = np.argsort(-d)[: int(len(l) * fr)]; l[a_] = 10**6 + np.arange(len(a_)); return sc(gt, l)
def one(f):
    s = load_subset(f, FW, FC); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); out = {}
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    A = normalize(TfidfVectorizer(max_features=100000, min_df=2, max_df=.5, sublinear_tf=True, stop_words="english", ngram_range=(1, 2), dtype=np.float32).fit_transform(T))
    B = normalize(TfidfVectorizer(max_features=100000, min_df=2, max_df=.5, sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32).fit_transform(T))
    L = fixz(TruncatedSVD(192, random_state=0).fit_transform(hstack([A, B]).tocsr()))
    out["Z"] = clus(Z, gt, arx); out["LSA"] = clus(L, gt, arx)
    for w in (.3, .5, .8): out[f"Z+{w}LSA"] = clus(normalize(np.hstack([Z, w * L])), gt, arx)
    Zc = Z - Z.mean(0); out["Z-mean"] = clus(fixz(Zc), gt, arx)
    u = TruncatedSVD(1, random_state=0).fit(Zc); out["Z-pc1"] = clus(fixz(Zc - u.inverse_transform(u.transform(Zc))), gt, arx)
    # retrofit: batch vocabulary vectors = idf-weighted mean of doc embeddings; docs re-embedded from them
    M = TfidfVectorizer(min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32).fit_transform(T)
    Wv = normalize(M.T @ Z); R = fixz(M @ Wv)
    for w in (.5, 1.0): out[f"Z+{w}retro"] = clus(normalize(np.hstack([Z, w * R])), gt, arx)
    return arx, out
if __name__ == "__main__":
    rounds = [int(x) for x in sys.argv[4:]] or [69, 70, 71, 72, 73]
    with ProcessPoolExecutor(5) as ex: R = list(ex.map(one, files_for(rounds)))
    for k in R[0][1]:
        so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); print("%-12s social %.4f arxiv %.4f round %.4f" % (k, so, ar, (3 * so + ar) / 4))
