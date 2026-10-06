"""arXiv-specialist table: teacher-neighbour recall and scores under both clusterers (unseen arXiv subsets 69-73)."""
import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from grid2 import fixz
from lab import sc
from arx_regime import mine, uh
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor
WP, FW, FC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def one(f):
    s = load_subset(f, FW, FC, True); gt = s["y"]; Z = fixz(s["X"] @ np.load(WP)); T = normalize(np.load("cache/" + os.path.basename(f).replace(".parquet", ".mpnet.npy")))
    a = NearestNeighbors(n_neighbors=16, metric="cosine").fit(Z).kneighbors(Z)[1]; b = NearestNeighbors(n_neighbors=16, metric="cosine").fit(T).kneighbors(T)[1]
    return dict(rec15=np.mean([len(set(p[1:]) & set(q[1:])) / 15 for p, q in zip(a, b)]), mine=sc(gt, mine(Z)), uh_noise=sc(gt, uh(Z, 8, False)), uh_single=sc(gt, uh(Z, 8, True)), uh_noise_seed2=sc(gt, uh(Z, 8, False, seed=2)))
if __name__ == "__main__":
    fs = [f for f in files_for([69, 70, 71, 72, 73]) if "arxiv" in f]
    with ProcessPoolExecutor(5) as ex: R = list(ex.map(one, fs))
    print(os.path.basename(WP), " ".join("%s %.4f" % (k, np.mean([r[k] for r in R])) for k in R[0]), flush=True)
