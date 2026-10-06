"""Self-training refinement: initial clusters -> linear classifier on exact batch TF-IDF -> recluster."""
import sys, numpy as np, pandas as pd, warnings, time
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
WP = sys.argv[1]
def final(Z, gt, arx, S=None):
    k, S0, fr = (35, 80, .4) if arx else (20, 120, .3); S = S or S0
    Z = smooth(Z, k, .4, 4); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)[0][:, -1]
    l = fcluster(linkage(Z, "average", "cosine"), S, "maxclust"); a_ = np.argsort(-d)[: int(len(l) * fr)]; core = np.ones(len(l), bool); core[a_] = False
    l2 = l.copy(); l2[a_] = 10**6 + np.arange(len(a_)); return sc(gt, l2), l, core, Z
def one(f):
    s = load_subset(f, 8192, 4096); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); out = {}
    base, l, core, Zs = final(Z, gt, arx); out["base"] = base
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    A = TfidfVectorizer(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, ngram_range=(1, 2), dtype=np.float32).fit_transform(T)
    B = TfidfVectorizer(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32).fit_transform(T)
    X = normalize(hstack([A, B]).tocsr())
    t0 = time.time()
    for Sp in (30, 60):
        lp = fcluster(linkage(Zs, "average", "cosine"), Sp, "maxclust"); ids, inv, cnt = np.unique(lp, return_inverse=True, return_counts=True)
        tr = core & (cnt[inv] >= 10)
        for alpha in (1.0, 10.0):
            clf = RidgeClassifier(alpha=alpha).fit(X[tr], lp[tr]); P = clf.decision_function(X); P = fixz(P - P.mean(1, keepdims=True))
            out[f"P S{Sp} a{alpha}"] = final(P, gt, arx)[0]
            for w in (.5, 1.0): out[f"Z+{w}P S{Sp} a{alpha}"] = final(normalize(np.hstack([Z, w * P])), gt, arx)[0]
    out["sec"] = time.time() - t0
    return arx, out
if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    for k in R[0][1]:
        so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); print("%-22s social %.4f arxiv %.4f round %.4f" % (k, so, ar, (3 * so + ar) / 4))
