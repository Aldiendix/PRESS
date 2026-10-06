"""PRESS text-clustering miner (Apex / Bittensor SN1, competition 10).

Pipeline: hashed word+char n-grams -> packed ternary student table (trained to imitate the
all-mpnet-base-v2 + UMAP + HDBSCAN ground truth) -> k-NN smoothing -> average-linkage clustering ->
low-density points returned as singletons.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, lzma, re
import numpy as np, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import diags, hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

FW, FC, D, IDF, ROWS, RICE, MARK = __FW__, __FC__, __D__, __IDF__, __ROWS__, __RICE__, __MARK__
# (smoothing k, alpha, iterations, clusters, singleton share) for social posts / paper titles
P_SOCIAL, P_TITLE = __P_SOCIAL__, __P_TITLE__
MAXN = 7000
TABLE = "__TABLE__"
_W = None
_URL = re.compile(r"https?://\S+|www\.\S+")
_MEN = re.compile(r"@\w+")
_ENT = re.compile(r"&\w+;")
_PUN = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


class ClusterRequest(BaseModel):
    texts: list[str]


class ClusterResponse(BaseModel):
    cluster_ids: list[int]


def prep(t):
    if MARK:  # keep link / mention markers and punctuation as signals
        t = _MEN.sub(" usr ", _URL.sub(" url ", t)).lower()
        return _WS.sub(" ", _ENT.sub(" ", t)).strip()[:1000]
    t = _MEN.sub(" ", _URL.sub(" ", t)).lower()
    return _WS.sub(" ", _PUN.sub(" ", _ENT.sub(" ", t))).strip()[:1000]


def table():
    """Decode TABLE (15 bits per character) into the float32 student matrix.

    Layout: nnz, unary-length | Rice-coded gaps between non-zero weights (unary quotients, then RICE-bit
    remainders) | signs | float16 column scales | LZMA'd int8 row-scale exponents (half-octaves).
    """
    global _W
    if _W is None:
        s = TABLE
        ln = (ord(s[0]) - 19968) << 15 | (ord(s[1]) - 19968)
        n = 0
        for c in s[2:]:
            n = n << 15 | (ord(c) - 19968)
        raw = (n >> ((len(s) - 2) * 15 - ln * 8)).to_bytes(ln, "big")
        k, ul = int.from_bytes(raw[:4], "big"), int.from_bytes(raw[4:8], "big")
        o = 8
        zeros = np.flatnonzero(np.unpackbits(np.frombuffer(raw[o:o + ul], np.uint8)) == 0)[:k]
        quo = np.diff(zeros, prepend=-1) - 1
        o += ul
        rb = (k * RICE + 7) // 8
        rem = np.unpackbits(np.frombuffer(raw[o:o + rb], np.uint8))[: k * RICE].reshape(k, RICE) @ (1 << np.arange(RICE - 1, -1, -1))
        o += rb
        pos = np.cumsum(quo * (1 << RICE) + rem + 1) - 1
        sb = (k + 7) // 8
        sign = np.unpackbits(np.frombuffer(raw[o:o + sb], np.uint8))[:k]
        o += sb
        F = FW + FC
        w = np.zeros(F * D, np.float32)
        w[pos] = 1.0 - 2.0 * sign
        w = w.reshape(F, D) * np.frombuffer(raw[o:o + 2 * D], np.float16).astype(np.float32)
        if ROWS:
            r = np.frombuffer(lzma.decompress(raw[o + 2 * D:]), np.int8).astype(np.float32)
            w *= np.exp2(r / 2)[:, None]
        _W = w
    return _W


def embed(docs):
    hw = HashingVectorizer(n_features=FW, ngram_range=(1, 2), analyzer="word", token_pattern=r"(?u)\b\w\w+\b|[^\w\s]" if MARK else r"(?u)\b\w\w+\b", alternate_sign=False, norm=None, dtype=np.float32)
    hc = HashingVectorizer(n_features=FC, ngram_range=(3, 5), analyzer="char_wb", alternate_sign=False, norm=None, dtype=np.float32)
    X = hstack([hw.transform(docs), hc.transform(docs)]).tocsr()
    X.data = np.log1p(X.data)
    if IDF:
        df = np.asarray((X > 0).sum(0)).ravel()
        X = X @ diags(np.log((X.shape[0] + 1.0) / (df + 1.0)) + 1.0)
    Z = np.asarray(normalize(X) @ table(), np.float32)
    z = np.abs(Z).sum(1) < 1e-9
    if z.any():
        Z[z] = np.random.RandomState(0).normal(size=(int(z.sum()), D)) * 1e-3
    return normalize(Z)


def smooth(Z, k, alpha, iters):
    ix = NearestNeighbors(n_neighbors=min(k + 1, len(Z)), metric="cosine").fit(Z).kneighbors(Z)[1][:, 1:]
    for _ in range(iters):
        Z = normalize((1 - alpha) * Z + alpha * Z[ix].mean(1))
    return Z


def core(Z, p):
    k, alpha, iters, S, fr = p
    n = len(Z)
    Z = smooth(Z, k, alpha, iters)
    d = NearestNeighbors(n_neighbors=min(16, n), metric="cosine").fit(Z).kneighbors(Z)[0][:, -1]
    lab = fcluster(linkage(Z, "average", "cosine"), max(2, min(S, n // 8)), "maxclust").astype(np.int64)
    out = np.argsort(-d)[: int(n * fr)]
    lab[out] = 10**6 + np.arange(len(out))
    return lab


def fallback(docs):
    X = TfidfVectorizer(max_features=50000, sublinear_tf=True, dtype=np.float32).fit_transform(docs)
    return MiniBatchKMeans(max(2, min(60, len(docs) // 40)), random_state=0, n_init=3).fit_predict(X)


def cluster_texts(texts):
    n = len(texts)
    if n < 3:
        return list(range(n))
    docs = [prep(t if isinstance(t, str) else str(t)) for t in texts]
    keep = np.array([len(d) > 0 for d in docs])
    if keep.sum() < 3:
        return [0] * n
    sub = [d for d, m in zip(docs, keep) if m]
    raw = [t for t, m in zip(texts, keep) if m]
    try:
        title = (np.median([len(d) for d in sub]) < 130 and np.mean(["\n" in t for t in raw]) < 0.08
                 and np.mean(["http" in t.lower() for t in raw]) < 0.03)
        p = P_TITLE if title else P_SOCIAL
        Z = embed(sub)
        m = len(sub)
        if m <= MAXN:
            lab = core(Z, p)
        else:  # cluster a sample, give the rest the majority label of their 5 nearest sampled points
            pick = np.sort(np.random.default_rng(0).choice(m, MAXN, replace=False))
            base = core(Z[pick], p)
            ix = NearestNeighbors(n_neighbors=5, metric="cosine").fit(Z[pick]).kneighbors(Z)[1]
            lab = np.array([np.bincount(r).argmax() for r in np.unique(base, return_inverse=True)[1][ix]])
    except Exception:
        lab = fallback(sub)
    lab = np.unique(lab, return_inverse=True)[1]
    res = np.full(n, -1, np.int64)
    res[keep] = lab
    return [int(x) for x in res]


def make_app():
    app = FastAPI(title="Text Clustering Miner")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/cluster", response_model=ClusterResponse)
    def cluster(request: ClusterRequest) -> ClusterResponse:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        return ClusterResponse(cluster_ids=cluster_texts(request.texts))

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    a = ap.parse_args()
    table()
    uvicorn.run(make_app(), host=a.host, port=a.port, log_level="info")
