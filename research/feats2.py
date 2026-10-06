"""Variant features: keep URL / mention markers and symbol tokens. Cached under cache/feat2/."""
import glob, os, re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import HashingVectorizer
URL = re.compile(r"https?://\S+|www\.\S+"); MEN = re.compile(r"@\w+"); ENT = re.compile(r"&\w+;"); WS = re.compile(r"\s+")
def prep(t):
    t = MEN.sub(" usr ", URL.sub(" url ", t)).lower()
    return WS.sub(" ", ENT.sub(" ", t)).strip()[:1000]
HW = HashingVectorizer(n_features=2**20, ngram_range=(1, 2), analyzer="word", token_pattern=r"(?u)\b\w\w+\b|[^\w\s]", alternate_sign=False, norm=None, dtype=np.float32)
HC = HashingVectorizer(n_features=2**20, ngram_range=(3, 5), analyzer="char_wb", alternate_sign=False, norm=None, dtype=np.float32)
def one(f):
    out = "cache/feat2/" + os.path.basename(f).replace(".parquet", "")
    if os.path.exists(out + ".c.npz"): return f
    t = [prep(x) for x in pd.read_parquet(f).text]
    sp.save_npz(out + ".w.npz", HW.transform(t)); sp.save_npz(out + ".c.npz", HC.transform(t)); return f
if __name__ == "__main__":
    os.makedirs("cache/feat2", exist_ok=True)
    with ProcessPoolExecutor(2) as ex:
        for i, f in enumerate(ex.map(one, sorted(glob.glob("data/round_*/*.parquet")))):
            if i % 60 == 0: print(i, f, flush=True)
