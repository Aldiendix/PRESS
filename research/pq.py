"""Post-hoc product quantization of a float student table; reports size and held-out quality."""
import sys, numpy as np, warnings, glob
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from evalw import quick_scores
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor

def feat_weight(fw, fc, n=24):
    fs = files_for(range(40, 69))[:: max(1, len(files_for(range(40, 69))) // n)]
    w = sum(np.asarray((load_subset(f, fw, fc)["X"] > 0).sum(0)).ravel() for f in fs).astype(np.float64)
    return w + 0.5

def pq(W, M, K, w, seed=0):
    d = W.shape[1] // M; out = np.zeros_like(W); codes = []
    for m in range(M):
        sub = W[:, m * d:(m + 1) * d]; km = KMeans(K, n_init=2, random_state=seed, max_iter=60).fit(sub, sample_weight=w)
        out[:, m * d:(m + 1) * d] = km.cluster_centers_[km.labels_]; codes.append(km.labels_)
    return out, np.array(codes)

def ev(args):
    f, Wq, fw, fc = args; s = load_subset(f, fw, fc); r = quick_scores(s["X"] @ Wq, s["y"], None); r["arxiv"] = s["arxiv"]; return r

if __name__ == "__main__":
    name, fw, fc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); W = np.load(f"cache/W_{name}.npy"); w = feat_weight(fw, fc)
    for M, K in [tuple(map(int, x.split("x"))) for x in sys.argv[4:]]:
        Wq, codes = pq(W, M, K, w); kb = (W.shape[0] * M * np.log2(K) / 8 + M * K * (W.shape[1] // M)) / 1024
        with ProcessPoolExecutor(4) as ex: R = list(ex.map(ev, [(f, Wq, fw, fc) for f in files_for([71, 72, 73])]))
        so = [r for r in R if not r["arxiv"]]; ar = [r for r in R if r["arxiv"]]
        print(f"{name} PQ M={M} K={K} ~{kb:.0f}KB | social agg70 %.4f pur %.4f | arxiv agg70 %.4f pur %.4f" % (np.mean([r['agg70'] for r in so]), np.mean([r['knn_pur'] for r in so]), np.mean([r['agg70'] for r in ar]), np.mean([r['knn_pur'] for r in ar])), flush=True)
        np.save(f"cache/W_{name}_pq{M}x{K}.npy", Wq)
