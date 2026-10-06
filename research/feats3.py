"""Feature-variant cache. Usage: feats3.py OUTDIR WORD_MAX CHAR_ANALYZER CHAR_MIN CHAR_MAX TRUNC  (rounds >= 40 only)"""
import glob, os, re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import HashingVectorizer
OUT, WMAX, CAN, CMIN, CMAX, TR = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
URL = re.compile(r"https?://\S+|www\.\S+"); MEN = re.compile(r"@\w+"); ENT = re.compile(r"&\w+;"); PUN = re.compile(r"[^\w\s]"); WS = re.compile(r"\s+")
def prep(t):
    t = URL.sub(" ", t); t = MEN.sub(" ", t).lower(); t = ENT.sub(" ", t); t = PUN.sub(" ", t)
    return WS.sub(" ", t).strip()[:TR]
HW = HashingVectorizer(n_features=2**20, ngram_range=(1, WMAX), analyzer="word", alternate_sign=False, norm=None, dtype=np.float32)
HC = HashingVectorizer(n_features=2**20, ngram_range=(CMIN, CMAX), analyzer=CAN, alternate_sign=False, norm=None, dtype=np.float32)
def one(f):
    out = OUT + "/" + os.path.basename(f).replace(".parquet", "")
    if os.path.exists(out + ".c.npz"): return f
    t = [prep(x) for x in pd.read_parquet(f).text]
    sp.save_npz(out + ".w.npz", HW.transform(t)); sp.save_npz(out + ".c.npz", HC.transform(t)); return f
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fs = [f for f in sorted(glob.glob("data/round_00*/*.parquet")) if int(f.split("round_")[1][:4]) >= 40]
    with ProcessPoolExecutor(int(os.environ.get("J", "5"))) as ex: list(ex.map(one, fs))
    print(OUT, len(fs), flush=True)
