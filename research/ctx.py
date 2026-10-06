"""Cross-call context: use texts already seen in earlier /cluster calls of the same round as extra (unlabelled) context."""
import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def pipeline(Zt, Tt, Zc=None, Tc=None, arx=False, kscale=1.0, joint_cls=True):
    """Cluster target rows (Zt, texts Tt); optional context rows (Zc, Tc) join smoothing and self-training only."""
    k, S, fr = PAR[arx]; n = len(Zt); Z = Zt if Zc is None else np.vstack([Zt, Zc]); T = Tt if Tc is None else Tt + Tc; kk = int(round(k * kscale))
    def stage(Zall):
        Zs = smooth(Zall, kk)[:n]; d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; return Zs, d, linkage(Zs, "average", "cosine")
    Zs, d, L = stage(Z); out = np.argsort(-d)[: int(n * fr)]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    kw = dict(max_features=60000, min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32)
    X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(T), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(T)]).tocsr())
    P = RidgeClassifier(alpha=10.0).fit(X[:n][ok], ps[ok]).decision_function(X); P = normalize(P - P.mean(1, keepdims=True))
    Zs, d, L = stage(normalize(np.hstack([Z, .75 * P]))); l = fcluster(L, S, "maxclust"); o = np.argsort(-d)[: int(n * fr)]; l[o] = 10**6 + np.arange(len(o)); return l

def run(args):
    grp, variant = args; res = []
    for i, c in enumerate(grp):       # grp = social subsets of one round, in call order
        if variant == "none" or i == 0: l = pipeline(c["Z"], c["T"])
        else:
            Zc = np.vstack([g["Z"] for g in grp[:i]]); Tc = sum([g["T"] for g in grp[:i]], [])
            l = pipeline(c["Z"], c["T"], Zc, Tc, kscale={"ctx": 1.0, "ctx_k": (len(Zc) + len(c["Z"])) / len(c["Z"]), "ctx_k15": 1.5}[variant])
        res.append(sc(c["gt"], l))
    return res
if __name__ == "__main__":
    for name in ("A", "B", "C"):
        D = [c for c in load(name) if not c["arx"]]; rounds = sorted(set(c["f"].split("round_")[1][:4] for c in D)); groups = [[c for c in D if "round_" + r in c["f"]] for r in rounds]
        for variant in ("none", "ctx", "ctx_k15", "ctx_k"):
            with ProcessPoolExecutor(5) as ex: R = np.array(list(ex.map(run, [(g, variant) for g in groups])))
            print("set %s %-8s social mean %.4f | by call position: 1st %.4f  2nd %.4f  3rd %.4f" % (name, variant, R.mean(), *R.mean(0)), flush=True)
