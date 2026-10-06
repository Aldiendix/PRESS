import sys, numpy as np; sys.path.insert(0, "research")
from lab import *
import igraph as ig, leidenalg
from scipy.sparse import csr_matrix

def knn_full(Zs, k):
    S = Zs @ Zs.T; np.fill_diagonal(S, -2); ix = np.argpartition(-S, k, axis=1)[:, :k]; sv = np.take_along_axis(S, ix, 1); o = np.argsort(-sv, 1)
    return np.take_along_axis(ix, o, 1), np.take_along_axis(sv, o, 1), S

def fn(c):
    k, S_, fr = PAR[c["arx"]]; ZA = c["ZA"]; n = len(ZA); m = int(n * fr); out = {}
    Zs = smooth(ZA, k); ix, sv, S = knn_full(Zs, 30); d = 1 - sv[:, 14]; o = np.argsort(-d)[:m]
    def single(l, idx=o): l = np.asarray(l).copy(); l[idx] = 10**6 + np.arange(len(idx)); return l
    D = 1 - S; np.fill_diagonal(D, 0)
    # --- 1. density peaks (Rodriguez & Laio, Science 2014)
    rho = -sv[:, :15].mean(1) * -1; rho = sv[:, :15].mean(1)      # density = mean similarity to 15 NN
    order = np.argsort(-rho); delta = np.zeros(n); parent = np.full(n, -1)
    for r, i in enumerate(order):
        if r == 0: delta[i] = D[i].max(); continue
        hi = order[:r]; j = hi[np.argmin(D[i, hi])]; delta[i] = D[i, j]; parent[i] = j
    gamma = (rho - rho.min()) * delta
    for nc in (40, 80, 120):
        cen = np.argsort(-gamma)[:nc]; lab = np.full(n, -1); lab[cen] = np.arange(nc)
        for i in order:
            if lab[i] < 0: lab[i] = lab[parent[i]] if parent[i] >= 0 else 0
        out[f"density peaks {nc} + knn singletons"] = single(lab)
        # halo: points whose density is below the highest border density of their cluster
        nb_other = lab[ix[:, :15]] != lab[:, None]; border = nb_other.any(1); halo = np.zeros(n, bool)
        for g in range(nc):
            mk = lab == g; bd = mk & border
            if bd.any(): halo |= mk & (rho < rho[bd].max())
        out[f"density peaks {nc} + halo singletons"] = single(lab, np.flatnonzero(halo))
    # --- 2. PhenoGraph: Jaccard SNN graph + Leiden (Levine et al. Cell 2015; Traag et al. Sci Rep 2019)
    for kk in (15, 30):
        A = np.zeros((n, n), np.float32); np.put_along_axis(A, ix[:, :kk], 1, 1); inter = A @ A.T; J = inter / (2 * kk - inter); J = J * ((A + A.T) > 0); r_, c_ = np.nonzero(np.triu(J, 1))
        g = ig.Graph(n=n, edges=list(zip(r_.tolist(), c_.tolist()))); g.es["weight"] = J[r_, c_].tolist()
        for res in (1.0, 3.0):
            lab = np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=res, seed=0).membership)
            out[f"jaccard-SNN k{kk} leiden res{res}"] = single(lab)
    # --- 3. hubness reduction: CSLS / local scaling for the linkage distances and for the neighbour graph
    r = sv[:, :10].mean(1); Dc = 1 - (2 * S - r[:, None] - r[None, :]) / 2; np.fill_diagonal(Dc, 0); Dc = np.maximum(Dc, 0); iu = np.triu_indices(n, 1)
    out["average linkage on CSLS distance"] = single(fcluster(linkage(Dc[iu], "average"), S_, "maxclust"))
    sig = np.sqrt(np.maximum(1 - sv[:, 6], 1e-6)); Dl = np.maximum(D, 0) / (sig[:, None] * sig[None, :]); out["average linkage, local scaling"] = single(fcluster(linkage(Dl[iu], "average"), S_, "maxclust"))
    S0 = ZA @ ZA.T; np.fill_diagonal(S0, -2); r0 = np.sort(S0, 1)[:, -10:].mean(1); C0 = 2 * S0 - r0[:, None] - r0[None, :]; ixc = np.argpartition(-C0, k, axis=1)[:, :k]; Zc = ZA
    for _ in range(4): Zc = normalize(.6 * Zc + .4 * Zc[ixc].mean(1))
    dc = np.sort(1 - Zc @ Zc.T, 1)[:, 15]; out["CSLS neighbours for smoothing"] = single(fcluster(linkage(Zc, "average", "cosine"), S_, "maxclust"), np.argsort(-dc)[:m])
    return out
if __name__ == "__main__":
    evaluate(fn, jobs=8)
