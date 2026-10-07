"""Joint random search of pipeline settings on rounds 61-74 (table trained on 40-60).
Odd rounds = search half, even rounds = confirmation half."""
import sys, os, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeClassifier
from scipy.cluster.hierarchy import fcluster, linkage
from lab import sc, smooth
V4 = {False: dict(k=20, a=.4, it=4, Sp=90, al=10.0, w=.75, S=120, fr=.2, dk=15, k2=1.0, fr1=.2), True: dict(k=35, a=.4, it=4, Sp=90, al=10.0, w=.75, S=180, fr=.35, dk=15, k2=1.0, fr1=.35)}
def prep_one(f):
    from train import load_subset; from feats import prep; from grid2 import fixz
    from sklearn.feature_extraction.text import TfidfVectorizer; from scipy.sparse import hstack
    s = load_subset(f, 8192, 16384, True); T = [prep(t) for t in pd.read_parquet(f, columns=["text"]).text]
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    return dict(f=f, rnd=int(f.split("round_")[1][:4]), gt=s["y"], arx=s["arxiv"], Z=fixz(s["X"] @ np.load("cache/W_w60.npy")).astype(np.float32), X=X)
def load():
    if os.path.exists("cache/search_D.pkl"): return pickle.load(open("cache/search_D.pkl", "rb"))
    from train import files_for
    with ProcessPoolExecutor(12) as ex: D = list(ex.map(prep_one, files_for(range(61, 75))))
    pickle.dump(D, open("cache/search_D.pkl", "wb")); return D
def run(c, p):
    Z, X, n = c["Z"], c["X"], len(c["gt"])
    Zs = smooth(Z, p["k"], p["a"], p["it"]); d = NearestNeighbors(n_neighbors=31, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, p["dk"]]; L = linkage(Zs, "average", "cosine")
    out = np.argsort(-d)[: int(n * p["fr1"])]; ps = fcluster(L, p["Sp"], "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=p["al"]).fit(X[ok], ps[ok]).decision_function(X); ZA = normalize(np.hstack([Z, p["w"] * normalize(P - P.mean(1, keepdims=True))]))
    Zs = smooth(ZA, max(5, int(round(p["k"] * p["k2"]))), p["a"], p["it"]); d = NearestNeighbors(n_neighbors=31, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, p["dk"]]
    l = fcluster(linkage(Zs, "average", "cosine"), p["S"], "maxclust"); o = np.argsort(-d)[: int(n * p["fr"])]; new = 10**6 + np.arange(len(o))
    if p.get("rj"):
        keep = np.ones(n, bool); keep[o] = False; ids = np.unique(l[keep]); cen = normalize(np.stack([Zs[keep & (l == i)].mean(0) for i in ids])); sim = Zs[o] @ cen.T; back = sim.max(1) >= p["rj"]; new[back] = ids[sim.argmax(1)][back]
    l[o] = new; return sc(c["gt"], l)
D = None
def job(a):
    i, p = a; return run(D[i], p)
def sample(rng, arx):
    return dict(k=int(rng.choice([12, 16, 20, 26, 35, 50])), a=float(rng.choice([.3, .4, .5])), it=int(rng.choice([3, 4, 5, 6])), Sp=int(rng.choice([60, 90, 130])), al=float(rng.choice([5., 10., 20.])),
                w=float(rng.choice([.5, .75, 1.0])), S=int(rng.choice([80, 120, 160, 220] if not arx else [80, 120, 180, 260])), fr=float(rng.choice([.15, .2, .25, .3] if not arx else [.25, .3, .35, .4, .45])),
                dk=int(rng.choice([10, 15, 25])), k2=float(rng.choice([.7, 1.0, 1.5])), fr1=float(rng.choice([.15, .25, .35])))
if __name__ == "__main__":
    D = load(); N = int(sys.argv[1]) if len(sys.argv) > 1 else 60; rng = np.random.default_rng(0); res = {}
    for arx in (False, True):
        tune = [i for i, c in enumerate(D) if c["arx"] == arx and c["rnd"] % 2 == 1]; conf = [i for i, c in enumerate(D) if c["arx"] == arx and c["rnd"] % 2 == 0]
        cfgs = [V4[arx]] + [sample(rng, arx) for _ in range(N)]
        with ProcessPoolExecutor(14) as ex: s_t = np.array(list(ex.map(job, [(i, p) for p in cfgs for i in tune]))).reshape(len(cfgs), len(tune)).mean(1)
        top = np.argsort(-s_t)[:6]; top = [0] + [t for t in top if t != 0]
        with ProcessPoolExecutor(14) as ex: s_c = np.array(list(ex.map(job, [(i, cfgs[t]) for t in top for i in conf]))).reshape(len(top), len(conf)).mean(1)
        print("ARXIV" if arx else "SOCIAL", "subsets: search %d, confirm %d" % (len(tune), len(conf)))
        for t, c_ in zip(top, s_c): print("  %s search %.4f (%+.4f)  confirm %.4f (%+.4f)  %s" % ("v4  " if t == 0 else "cand", s_t[t], s_t[t] - s_t[0], c_, c_ - s_c[0], cfgs[t]), flush=True)
        res[arx] = (cfgs, s_t)
    pickle.dump(res, open("cache/search_res.pkl", "wb"))
