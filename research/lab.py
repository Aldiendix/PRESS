"""Test bench: cached student / self-trained embeddings for unseen sets; plug in a final-stage function.

Sets: A = table trained on rounds 40-68, tested on 69-73 (20 subsets); B = table trained on 40-73, tested on 74.
"""
import os, sys, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from scipy.cluster.hierarchy import fcluster, linkage
sc = lambda g, p: (max(0, ari(g, p)) + nmi(g, p)) / 2
SETS = {"A": ("cache/W_w2.npy", [69, 70, 71, 72, 73]), "B": ("cache/W_nU73.npy", [74]), "C": ("cache/W_nU72.npy", [73])}
PAR = {False: (20, 120, .2), True: (35, 180, .35)}   # k, clusters, singleton share (v4 settings)

def smooth(Z, k, a=.4, it=4):
    ix = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z).kneighbors(Z)[1][:, 1:]
    for _ in range(it): Z = normalize((1 - a) * Z + a * Z[ix].mean(1))
    return Z

def _prep(args):
    f, wp = args
    from train import load_subset; from feats import prep; from grid2 import fixz
    from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import RidgeClassifier; from scipy.sparse import hstack
    s = load_subset(f, 8192, 16384, True); gt = s["y"]; arx = s["arxiv"]; Z = fixz(s["X"] @ np.load(wp)); n = len(gt); k, S, fr = PAR[arx]
    T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    Zs = smooth(Z, k); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; L = linkage(Zs, "average", "cosine")
    out = np.argsort(-d)[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
    ZA = normalize(np.hstack([Z, .75 * P])).astype(np.float32)
    return dict(f=f, gt=gt, arx=arx, Z=Z.astype(np.float32), ZA=ZA, P=P.astype(np.float32), X=X, T=T)

def load(name):
    p = f"cache/lab_{name}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    from train import files_for
    wp, rounds = SETS[name]
    with ProcessPoolExecutor(10) as ex: D = list(ex.map(_prep, [(f, wp) for f in files_for(rounds)]))
    pickle.dump(D, open(p, "wb")); return D

def baseline(c):
    k, S, fr = PAR[c["arx"]]; Zs = smooth(c["ZA"], k); n = len(Zs)
    d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]
    l = fcluster(linkage(Zs, "average", "cosine"), S, "maxclust"); o = np.argsort(-d)[: int(n * fr)]; l[o] = 10**6 + np.arange(len(o)); return l

_FN = None
def _run(c):
    out = _FN(c); out = out if isinstance(out, dict) else {"": out}
    return c["arx"], {k: sc(c["gt"], v) for k, v in out.items()}

def evaluate(fn, sets=("A", "B"), jobs=10):
    """fn(ctx) -> labels or {variant: labels}. Prints round-level scores (3 social : 1 arXiv) per set."""
    global _FN; _FN = fn; res = {}
    for name in sets:
        D = load(name)
        with ProcessPoolExecutor(jobs) as ex: R = list(ex.map(_run, D))
        for k in R[0][1]:
            so = np.mean([o[k] for a, o in R if not a]); ar = np.mean([o[k] for a, o in R if a]); res.setdefault(k, []).append(((3 * so + ar) / 4, so, ar))
    for k, v in res.items(): print("%-34s" % k, " | ".join("%s %.4f (soc %.4f arx %.4f)" % ((n,) + x) for n, x in zip(sets, v)), flush=True)
    return res

if __name__ == "__main__":
    evaluate(baseline, sets=("A", "B", "C"))
