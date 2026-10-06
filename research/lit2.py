"""Evidence-accumulation ensemble (Fred & Jain 2005; SC3, Nat Methods 2017): consensus clusters + instability as a noise score.
Also out-of-fold classifier margin (confident learning) and small-cluster (HDBSCAN-style) noise candidates."""
import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
OUT = {}
def fn(c):
    k, S_, fr = PAR[c["arx"]]; ZA = c["ZA"]; n = len(ZA); gt = c["gt"]; y = gt == -1; rng = np.random.default_rng(0); out = {}
    base = baseline(c); out["baseline"] = base
    C = np.zeros((n, n), np.float32); B = 16
    for b in range(B):
        cols = rng.choice(ZA.shape[1], int(ZA.shape[1] * .7), replace=False); Zb = smooth(normalize(ZA[:, cols]), int(rng.choice([10, 15, 20, 30])))
        l = fcluster(linkage(Zb, "average", "cosine"), int(rng.integers(25, 46)), "maxclust"); oh = np.zeros((n, l.max() + 1), np.float32); oh[np.arange(n), l] = 1; C += oh @ oh.T
    C /= B; iu = np.triu_indices(n, 1)
    Lc = linkage((1 - C)[iu], "average"); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(smooth(ZA, k)).kneighbors()[0][:, -1] if False else None
    Zs = smooth(ZA, k); dk = np.sort(1 - Zs @ Zs.T, 1)[:, 15]; o = np.argsort(-dk)[: int(n * fr)]
    def single(l, idx=o): l = np.asarray(l).copy(); l[idx] = 10**6 + np.arange(len(idx)); return l
    for K, nm in ((35, "35"), (60, "60"), (S_, "S")): out[f"consensus linkage K={nm} + knn singletons"] = single(fcluster(Lc, K, "maxclust"))
    lc = fcluster(Lc, 35, "maxclust"); oh = np.zeros((n, lc.max() + 1), np.float32); oh[np.arange(n), lc] = 1; cnt = oh.sum(0)
    own = ((C @ oh)[np.arange(n), lc] - 1) / np.maximum(cnt[lc] - 1, 1); instab = 1 - own
    srt = np.sort(C @ oh / cnt, 1); amb = -(srt[:, -1] - srt[:, -2])                      # margin between best and second-best consensus cluster
    out[f"consensus K=120, singletons by instability"] = single(fcluster(Lc, S_, "maxclust"), np.argsort(-instab)[: int(n * fr)])
    out[f"baseline clusters, singletons by instability"] = single(fcluster(linkage(Zs, "average", "cosine"), S_, "maxclust"), np.argsort(-instab)[: int(n * fr)])
    # out-of-fold ridge margin at coarse granularity
    X = c["X"]; lab35 = fcluster(linkage(Zs, "average", "cosine"), 35, "maxclust"); big = np.bincount(lab35)[lab35] >= 10; marg = np.zeros(n)
    yb = lab35[big]; Xb = X[big]; idxb = np.flatnonzero(big); sco = np.zeros((n, lab35.max() + 1), np.float32)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xb, yb):
        clf = RidgeClassifier(alpha=10.0).fit(Xb[tr], yb[tr]); sco[np.ix_(idxb[te], clf.classes_)] = clf.decision_function(Xb[te])
    if (~big).any(): clf = RidgeClassifier(alpha=10.0).fit(Xb, yb); sco[np.ix_(np.flatnonzero(~big), clf.classes_)] = clf.decision_function(X[~big])
    s2 = np.sort(sco, 1); oof = -(s2[:, -1] - s2[:, -2]); selfc = -sco[np.arange(n), lab35]
    l120 = fcluster(linkage(Zs, "average", "cosine"), S_, "maxclust"); small = -np.bincount(l120)[l120].astype(float)
    rk = lambda v: np.argsort(np.argsort(v)) / n
    sig = dict(instab=instab, amb=amb, oof_margin=oof, oof_self=selfc, small=small, knn=dk, combo=rk(instab) + rk(oof) + rk(dk), combo2=rk(instab) + rk(selfc))
    for nm, v in sig.items():
        od = np.argsort(-v)
        for q in (.05, .15):
            top = od[: int(n * q)]; OUTK = f"prec {nm} @{q}"; out[OUTK] = ("prec", float(y[top].mean()))
            l = base.copy(); l[top] = -1; out[f"shared -1: {nm} top {q}"] = l
    return out
import lab
def _run2(c):
    o = fn(c); return c["arx"], {k: (v[1] if isinstance(v, tuple) else sc(c["gt"], v)) for k, v in o.items()}
if __name__ == "__main__":
    for name in ("A", "B"):
        D = load(name)
        with ProcessPoolExecutor(7) as ex: R = list(ex.map(_run2, D))
        print("== set", name, " noise base rate %.2f" % np.mean([(c["gt"] == -1).mean() for c in D]))
        for kk in R[0][1]:
            so = np.mean([o[kk] for a, o in R if not a]); ar = np.mean([o[kk] for a, o in R if a]); print("%-48s %.4f (soc %.4f arx %.4f)" % (kk, (3 * so + ar) / 4, so, ar), flush=True)
