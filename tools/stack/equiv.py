"""Exact equality of cluster ids: reference file vs candidate files on whole round subsets.
usage: equiv.py <ref.py> <cand1.py,cand2.py> <round> [<round> ...]"""
import sys, os, time, glob, importlib.util, resource
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
def load(p):
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ref = load(sys.argv[1]); cands = [(os.path.basename(p), load(p)) for p in sys.argv[2].split(",")]
Wr = ref.table(); print("tables equal:", [bool(np.array_equal(Wr, m.table())) for _, m in cands], flush=True)
bad = 0; tt = {n: [] for n, _ in cands}; tr = []
for rnd in sys.argv[3:]:
    for f in sorted(glob.glob("/root/PRESS/data/round_%04d/*.parquet" % int(rnd))):
        x = pd.read_parquet(f).text.tolist(); t = time.time(); a = ref.cluster_texts(x); tr.append(time.time() - t); res = []
        for n, m in cands:
            t = time.time(); b = m.cluster_texts(x); tt[n].append(time.time() - t); res.append(a == b); bad += a != b
        print(os.path.basename(f)[:28], "identical", res, "clusters", len(set(a)), flush=True)
print("DIFFERENT" if bad else "ALL IDENTICAL", "| s/subset ref %.1f" % np.mean(tr), {n: round(float(np.mean(v)), 1) for n, v in tt.items()}, "| peak RSS MB %.0f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
