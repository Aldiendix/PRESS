import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
import umap, hdbscan
from train import load_subset, files_for
from grid2 import fixz
from lab import smooth, sc
from arx_regime import mine, uh
from sklearn.preprocessing import normalize
from concurrent.futures import ProcessPoolExecutor
def one(f):
    s = load_subset(f, 8192, 16384, True); gt = s["y"]; res = {}
    T = normalize(np.load("cache/" + os.path.basename(f).replace(".parquet", ".mpnet.npy"))); Zst = fixz(s["X"] @ np.load("cache/W_nU72.npy"))
    for nm, Z in (("teacher", T), ("ternary (trained <=72)", Zst)):
        for S, fr in ((120, .2), (40, .2), (120, .1), (40, 0.0)): res[f"{nm} | my pipeline S{S} fr{fr}"] = sc(gt, mine(Z, S, fr, 20))
        res[nm + " | umap+hdbscan noise=-1"] = sc(gt, uh(Z, 8, False))
    for w in (.3, .5, .7):   # how good must the embedding be? mix teacher and student similarity structure
        M = normalize(np.hstack([np.sqrt(w) * T, np.sqrt(1 - w) * Zst])); res[f"mix {w} teacher | my pipeline"] = sc(gt, mine(M, 120, .2, 20)); res[f"mix {w} teacher | umap+hdbscan -1"] = sc(gt, uh(M, 8, False))
    return res
if __name__ == "__main__":
    fs = [f for f in files_for([73]) if "arxiv" not in f]
    with ProcessPoolExecutor(3) as ex: R = list(ex.map(one, fs))
    for k in R[0]: print("%-48s %.4f" % (k, np.mean([r[k] for r in R])))
