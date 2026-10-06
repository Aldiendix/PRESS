"""Cache hashed n-gram count features (2^20 buckets, foldable by modulo) for every round parquet."""
import glob, os, re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import HashingVectorizer
URL = re.compile(r"https?://\S+|www\.\S+"); MEN = re.compile(r"@\w+"); ENT = re.compile(r"&\w+;"); PUN = re.compile(r"[^\w\s]"); WS = re.compile(r"\s+")
def prep(t):
    t = URL.sub(" ", t); t = MEN.sub(" ", t).lower(); t = ENT.sub(" ", t); t = PUN.sub(" ", t)
    return WS.sub(" ", t).strip()[:1000]
HW = HashingVectorizer(n_features=2**20, ngram_range=(1, 2), analyzer="word", alternate_sign=False, norm=None, dtype=np.float32)
HC = HashingVectorizer(n_features=2**20, ngram_range=(3, 5), analyzer="char_wb", alternate_sign=False, norm=None, dtype=np.float32)
def one(f):
    out = "cache/feat/" + os.path.basename(f).replace(".parquet", "")
    if os.path.exists(out + ".c.npz"): return f
    t = [prep(x) for x in pd.read_parquet(f).text]
    sp.save_npz(out + ".w.npz", HW.transform(t)); sp.save_npz(out + ".c.npz", HC.transform(t)); return f
if __name__ == "__main__":
    os.makedirs("cache/feat", exist_ok=True)
    fs = sorted(glob.glob("data/round_*/*.parquet"))
    with ProcessPoolExecutor(5) as ex:
        for i, f in enumerate(ex.map(one, fs)):
            if i % 40 == 0: print(i, f, flush=True)
