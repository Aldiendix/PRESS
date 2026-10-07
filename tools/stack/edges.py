"""Edge-case equality (sampling path, empty texts, tiny batches, non-str, link posts): reference file vs candidates.
usage: edges.py <ref.py> <cand1.py,cand2.py>"""
import os, sys, importlib.util, glob
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
def load(p):
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ref = load(sys.argv[1]); new = [load(p) for p in sys.argv[2].split(",")]
a = pd.read_parquet(glob.glob("/root/PRESS/data/round_0074/*subset_2*.parquet")[0]).text.tolist()
b = pd.read_parquet(glob.glob("/root/PRESS/data/round_0073/*subset_1*.parquet")[0]).text.tolist()
link = ["look at this https://preview.redd.it/abc%d.jpg?width=640" % i for i in range(30)]
cases = {"7000 texts (sample path)": a + b[:2000], "400 texts (no refine)": a[:400], "with empty/symbol-only": ["", "!!!", "   "] + a[:600] + ["@x http://t.co/a"],
         "0": [], "1": a[:1], "2": a[:2], "3": a[:3], "all empty": ["", "", "", "?"], "5 texts": a[:5], "40 texts": a[:40], "600 + 30 link posts": a[:600] + link,
         "1500 + 19 link posts": a[:1500] + link[:19], "non-str items": a[:700] + [None, 5, 3.5], "600 texts x2 (duplicates)": a[:600] * 2}
for k, t in cases.items():
    x = ref.cluster_texts(t); ys = [m.cluster_texts(t) for m in new]
    print("%-28s n=%5d identical=%s types_ok=%s ids=%d" % (k, len(t), [x == y for y in ys], all(type(v) is int for y in ys for v in y), len(set(x))), flush=True)
