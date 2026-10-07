"""Learned merging of fine clusters. Table trained on rounds 40-60; merge model trained on rounds 61-67, tested on 68-74."""
import sys, numpy as np, pickle, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
import search as S
from lab import sc, smooth
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeClassifier
from scipy.cluster.hierarchy import fcluster, linkage, cophenet
from scipy.spatial.distance import squareform
FINE = int(os.environ.get("FINE", "300")); NC = 10
FEATS = ["cos_smooth", "cos_raw", "cos_cls", "cos_tfidf", "link_ab", "link_ba", "avg_sim", "coph", "log_min", "log_max", "rank_ab", "rank_ba", "tight_a", "tight_b"]
def stage(i):
    c = S.D[i]; p = dict(S.V4[c["arx"]]); p.update(dict(S=80, fr=.3) if c["arx"] else {}); Z, X, gt, n = c["Z"], c["X"], c["gt"], len(c["gt"])
    Zs = smooth(Z, p["k"]); d = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]; L = linkage(Zs, "average", "cosine")
    out = np.argsort(-d)[: int(n * p["fr1"])]; ps = fcluster(L, 90, "maxclust"); ok = np.ones(n, bool); ok[out] = False; ok &= np.bincount(ps)[ps] >= 10
    P = RidgeClassifier(alpha=10.0).fit(X[ok], ps[ok]).decision_function(X); Pn = normalize(P - P.mean(1, keepdims=True)); ZA = normalize(np.hstack([Z, .75 * Pn]))
    Zs = smooth(ZA, p["k"]); dd, ix = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Zs).kneighbors(Zs); d = dd[:, -1]; L = linkage(Zs, "average", "cosine")
    o = np.argsort(-d)[: int(n * p["fr"])]; keep = np.ones(n, bool); keep[o] = False
    base = fcluster(L, p["S"], "maxclust"); base[o] = 10**6 + np.arange(len(o))
    lf = fcluster(L, FINE, "maxclust"); ids = np.unique(lf[keep]); K = len(ids); idx_of = {g: j for j, g in enumerate(ids)}; lab = np.array([idx_of.get(g, -1) for g in lf]); lab[~keep] = -1
    oh = np.zeros((n, K), np.float32); kk = np.flatnonzero(keep); oh[kk, lab[kk]] = 1; cnt = oh.sum(0)
    def cen(M): return normalize((oh.T @ M) / cnt[:, None])
    Cs, Cr, Cp = cen(Zs), cen(Z), cen(Pn); Cx = normalize(np.asarray((oh.T @ X) / cnt[:, None])); SS_ = Cs @ Cs.T
    # cross-neighbour links: share of A's 15-NN edges that land in B
    nb = lab[ix[:, 1:]]; link = np.zeros((K, K), np.float32)
    for a_ in range(15): m = keep & (nb[:, a_] >= 0); np.add.at(link, (lab[m], nb[m, a_]), 1)
    link /= (cnt[:, None] * 15)
    avg = (oh.T @ (Zs @ (Zs.T @ oh))) / (cnt[:, None] * cnt[None, :])   # mean pairwise cosine between members
    co = squareform(cophenet(L)); first = np.array([np.flatnonzero(lab == j)[0] for j in range(K)]); coph = co[np.ix_(first, first)]; del co
    tight = np.diag(avg).copy(); np.fill_diagonal(SS_, -2); order = np.argsort(-SS_, 1); rank = np.argsort(order, 1)
    maj = np.array([np.bincount(gt[lab == j] + 1).argmax() - 1 for j in range(K)]); pairs = set()
    for a_ in range(K):
        for b_ in order[a_, :NC]: pairs.add((min(a_, b_), max(a_, b_)))
    pairs = np.array(sorted(pairs)); A, B = pairs[:, 0], pairs[:, 1]
    F = np.column_stack([SS_[A, B], (Cr @ Cr.T)[A, B], (Cp @ Cp.T)[A, B], (Cx @ Cx.T)[A, B], np.maximum(link[A, B], link[B, A]), np.minimum(link[A, B], link[B, A]), avg[A, B], coph[A, B],
                         np.log(np.minimum(cnt[A], cnt[B])), np.log(np.maximum(cnt[A], cnt[B])), np.minimum(rank[A, B], rank[B, A]), np.maximum(rank[A, B], rank[B, A]), np.maximum(tight[A], tight[B]), np.minimum(tight[A], tight[B])])
    y = (maj[A] == maj[B]) & (maj[A] >= 0); w = np.sqrt(cnt[A] * cnt[B])
    return dict(arx=c["arx"], rnd=c["rnd"], gt=gt, base=base, lab=lab, o=o, K=K, pairs=pairs, F=F.astype(np.float32), y=y, w=w, cnt=cnt, coph=coph.astype(np.float32), maj=maj)
def merged(r, prob, t, link_="average"):
    K = r["K"]; Dm = np.ones((K, K), np.float32); A, B = r["pairs"][:, 0], r["pairs"][:, 1]; Dm[A, B] = Dm[B, A] = 1 - prob; np.fill_diagonal(Dm, 0)
    cl = fcluster(linkage(squareform(Dm, checks=False), link_), t, "distance"); l = np.where(r["lab"] >= 0, cl[np.maximum(r["lab"], 0)], 0).astype(np.int64); l[r["o"]] = 10**6 + np.arange(len(r["o"])); return l
if __name__ == "__main__":
    pk = f"cache/merge_{FINE}.pkl"
    if os.path.exists(pk): R = pickle.load(open(pk, "rb"))
    else:
        S.D = S.load()
        with ProcessPoolExecutor(10) as ex: R = list(ex.map(stage, range(len(S.D))))
        pickle.dump(R, open(pk, "wb"))
    from sklearn.ensemble import HistGradientBoostingClassifier; from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score
    for arx in (False, True):
        tr = [r for r in R if r["arx"] == arx and r["rnd"] <= 67]; te = [r for r in R if r["arx"] == arx and r["rnd"] >= 68]
        Xtr = np.vstack([r["F"] for r in tr]); ytr = np.concatenate([r["y"] for r in tr]); wtr = np.concatenate([r["w"] for r in tr])
        print("ARXIV" if arx else "SOCIAL", "train subsets %d (pairs %d, positive %.2f), test subsets %d, fine clusters ~%d" % (len(tr), len(ytr), ytr.mean(), len(te), np.mean([r["K"] for r in te])))
        base = np.mean([sc(r["gt"], r["base"]) for r in te]); print("   baseline (uniform cut) on test: %.4f" % base)
        orc = np.mean([sc(r["gt"], merged(r, r["y"].astype(np.float32), .5)) for r in te]); print("   oracle pair labels -> merged: %.4f" % orc)
        models = {"GBM": HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=.08), "logistic": None}
        for name in models:
            if name == "GBM": m = models[name].fit(Xtr, ytr, sample_weight=wtr); pf = lambda F: m.predict_proba(F)[:, 1]
            else:
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; lr = LogisticRegression(C=1.0, max_iter=500).fit((Xtr - mu) / sd, ytr, sample_weight=wtr); pf = lambda F: lr.predict_proba((F - mu) / sd)[:, 1]
                print("   logistic weights:", " ".join("%s %+.2f" % (f, w_) for f, w_ in zip(FEATS, lr.coef_[0])))
            auc = np.mean([roc_auc_score(r["y"], pf(r["F"])) for r in te if 0 < r["y"].mean() < 1]); line = []
            for lk in ("average", "single", "complete"):
                for t in (.3, .4, .5, .6, .7, .8):
                    trs = np.mean([sc(r["gt"], merged(r, pf(r["F"]), t, lk)) for r in tr]); tes = np.mean([sc(r["gt"], merged(r, pf(r["F"]), t, lk)) for r in te]); line.append((trs, tes, lk, t))
            b = max(line); print("   %-8s pair AUC %.3f | chosen on train: %s t=%.1f -> train %.4f, TEST %.4f (%+.4f vs baseline) | best test in grid %.4f" % (name, auc, b[2], b[3], b[0], b[1], b[1] - base, max(x[1] for x in line)))
