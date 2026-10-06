"""Upper bound for 'noise as a property of the title': many teacher-pipeline runs over a small pool."""
import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
NP, NS = 30000, 120
P = pd.read_parquet("cache/arxiv_pool.parquet").iloc[:240000]; pool = np.flatnonzero(P.id.str[:2].values == "26")[:NP]
def one(r):
    import umap, hdbscan
    rng = np.random.default_rng(1000 + r); idx = np.sort(rng.choice(pool, 5000, replace=False))
    E = np.load("cache/arxiv_pool.f16.npy", mmap_mode="r")[idx].astype(np.float32)
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=r, n_jobs=1).fit_transform(E)
    return idx, hdbscan.HDBSCAN(min_cluster_size=25, min_samples=8).fit_predict(U)
if __name__ == "__main__":
    with ProcessPoolExecutor(14) as ex: R = list(ex.map(one, range(NS)))
    R = [r for r in R if r[1].max() + 1 >= 10]; print(len(R), "valid runs; noise rate %.3f" % np.mean([(l == -1).mean() for _, l in R]))
    import pickle; pickle.dump(R, open("cache/stab.pkl", "wb"))
    # leave-one-run-out: propensity from the other runs predicts noise in the held-out run
    cnt = np.zeros(240000); nz = np.zeros(240000)
    for idx, l in R: cnt[idx] += 1; nz[idx] += l == -1
    prec = {q: [] for q in (.05, .1, .2, .3)}; aucs = []
    from sklearn.metrics import roc_auc_score
    for idx, l in R:
        y = l == -1; c = cnt[idx] - 1; p = (nz[idx] - y) / np.maximum(c, 1); ok = c >= 5; y, p = y[ok], p[ok]; o = np.argsort(-p); aucs.append(roc_auc_score(y, p))
        for q in prec: prec[q].append(y[o[: int(len(y) * q)]].mean())
    print("oracle title propensity -> held-out run: AUC %.3f | precision at top 5%% %.2f, 10%% %.2f, 20%% %.2f, 30%% %.2f" % (np.mean(aucs), *[np.mean(prec[q]) for q in prec]))
    prop = nz[cnt >= 10] / cnt[cnt >= 10]; print("propensity distribution (titles with >=10 runs): deciles", np.round(np.quantile(prop, np.arange(.1, 1, .1)), 2), "| share always noise (>=0.9): %.3f, never (<=0.1): %.3f" % ((prop >= .9).mean(), (prop <= .1).mean()))
