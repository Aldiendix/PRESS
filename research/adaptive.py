"""Oracle headroom of per-subset cluster count / singleton share, and which batch statistics predict the best share."""
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
FRS = (.1, .15, .2, .25, .3, .35, .4, .5); SS = (40, 60, 80, 120, 180)
def one(f):
    s = load_subset(f, FW, FC, True); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(WP)); n = len(gt)
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    k, fr0 = (35, .4) if arx else (20, .3)
    def st(Z):
        Zs = smooth(Z, k, .4, 4); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; return Zs, d, linkage(Zs, "average", "cosine")
    Zs, d, L = st(Z); out = np.argsort(-d)[: int(n * fr0)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
    Zs, d, L = st(normalize(np.hstack([Z, .75 * P]))); order = np.argsort(-d); res = {}
    for S in SS:
        l0 = fcluster(L, S, "maxclust")
        for fr in FRS:
            l = l0.copy(); o = order[: int(n * fr)]; l[o] = 10**6 + np.arange(len(o)); res[(S, fr)] = sc(gt, l)
    rnd = np.random.default_rng(0).choice(n, 1500, replace=False); rel = d.mean() / (1 - Zs[rnd] @ Zs[rnd].T)[np.triu_indices(1500, 1)].mean()
    stats = dict(rel=rel, dmean=d.mean(), dstd=d.std() / d.mean(), noise=(gt == -1).mean(), k=gt.max() + 1, medlen=np.median([len(t) for t in T]))
    return arx, res, stats, f[-40:-8]
if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, files_for(range(62, 74))))
    for grp in (False, True):
        sel = [r for r in R if r[0] == grp]; keys = sel[0][1].keys()
        fixed = max(keys, key=lambda k: np.mean([r[1][k] for r in sel])); print("ARXIV" if grp else "SOCIAL", "best fixed", fixed, "%.4f" % np.mean([r[1][fixed] for r in sel]),
              "| oracle per-subset %.4f" % np.mean([max(r[1].values()) for r in sel]), "| oracle fr only (S fixed) %.4f" % np.mean([max(r[1][(fixed[0], fr)] for fr in FRS) for r in sel]),
              "| oracle S only %.4f" % np.mean([max(r[1][(S, fixed[1])] for S in SS) for r in sel]))
        bf = np.array([max(FRS, key=lambda fr: r[1][(fixed[0], fr)]) for r in sel])
        for st_ in ("rel", "dmean", "dstd", "noise", "k", "medlen"): print("   corr(best fr, %s) = %.2f" % (st_, np.corrcoef(bf, [r[2][st_] for r in sel])[0, 1]))
    import pickle; pickle.dump(R, open("cache/adaptive.pkl", "wb"))
