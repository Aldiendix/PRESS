"""Self-training parameter / iteration study on a fixed table (rounds 69-73, unseen by the table)."""
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
WP, FW, FC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def stage(Z, arx):
    k, S, fr = (35, 80, .4) if arx else (20, 120, .3)
    Zs = smooth(Z, k, .4, 4); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]
    L = linkage(Zs, "average", "cosine"); out = np.argsort(-d)[: int(len(Z) * fr)]; return L, out, S


def final(L, out, S, gt):
    l = fcluster(L, S, "maxclust"); l[out] = 10**6 + np.arange(len(out)); return sc(gt, l)


def one(f):
    s = load_subset(f, FW, FC, True); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); res = {}
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    L0, out0, S = stage(Z, arx); res["base"] = final(L0, out0, S, gt)
    for Sp in (90,):
        for al in (10.0,):
            for w in (.75,):
                L, out = L0, out0
                for it in (1,):
                    ps = fcluster(L, Sp, "maxclust"); ok = np.ones(len(gt), bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
                    P = RidgeClassifier(alpha=al).fit(X[ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
                    L, out, _ = stage(normalize(np.hstack([Z, w * P])), arx); res[(Sp, al, w, it)] = final(L, out, S, gt)
                    if w != .75 and it == 1:
                        break
    return arx, res


if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, files_for([69, 70, 71, 72, 73])))
    keys = set.intersection(*[set(o) for _, o in R]); rows = []
    for k in keys:
        so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); rows.append(((3 * so + ar) / 4, so, ar, k))
    for r in sorted(rows, key=lambda r: -r[0])[:14] + [r for r in rows if r[3] == "base"]: print("round %.4f social %.4f arxiv %.4f" % r[:3], r[3])
