"""PRESS text-clustering miner (Apex / Bittensor SN1, competition 10).

Pipeline: hashed word+char n-grams -> packed ternary student table (trained to imitate the
all-mpnet-base-v2 + UMAP + HDBSCAN ground truth) -> k-NN smoothing -> average-linkage clustering ->
self-training on the batch's exact vocabulary -> the points a linear ranker expects to score better alone
(by default the lowest-density points) returned as singletons.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, re
from math import comb
import numpy as np, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import hstack, vstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import sklearn

sklearn.set_config(working_memory=96)

FW, FC, D, IDF, BITS = __FW__, __FC__, __D__, __IDF__, __BITS__
# (smoothing k, alpha, iterations, clusters, singleton share, 17 weights of the singleton ranker) for social posts /
# paper titles.  Without the 6th element the lowest-density points become the singletons (the v4.1 rule).
P_SOCIAL, P_TITLE = __P_SOCIAL__, __P_TITLE__
MAXN = 6000
# self-training: (first-pass clusters, ridge alpha, weight of classifier scores); None = off
REFINE = __REFINE__
# language split of social clusters (1 / 0); smallest group of Reddit image posts that gets one shared id (0 = off)
LANG, LINK = __LANG__, __LINK__
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
    t = _MEN.sub(" ", _URL.sub(" ", t)).lower()
    return _WS.sub(" ", _PUN.sub(" ", _ENT.sub(" ", t))).strip()[:1000]


def table():
    """Decode TABLE (BITS bits per character) into the float32 student matrix.

    One big integer read from the top: 16b number of row symbols | 14b frequency of each symbol (sum 2^14) |
    D x 16b float16 column scales | 8b (lowest row-scale level + 128) | per chunk of 256 rows: 20b length L and an
    L-bit ANS state.  A row symbol is 4 * (non-zero count) + (row-scale level - lowest level), 0 for an empty row;
    after the symbol the state holds the rank of the row's column set among C(D, count) and one sign bit per column.
    """
    global _W
    if _W is None:
        n = 0
        for c in TABLE:
            n = n << BITS | ord(c) - __BASE__  # the builder writes the alphabet base (and, for 19 bits, the XOR mask)
        o = len(TABLE) * BITS

        def G(w):
            nonlocal o
            o -= w
            return n >> o & (1 << w) - 1

        q = [G(14) for _ in range(G(16))]
        s = np.array([G(16) for _ in range(D)], np.uint16).view(np.float16).astype(np.float32)
        m = G(8) - 128
        y, u, a = [], [], 0
        for i, v in enumerate(q):
            u.append(a)
            y += [i] * v
            a += v
        B = [[comb(c, i) for c in range(D + 1)] for i in range(len(q) // 4 + 1)]
        F = FW + FC
        w = np.zeros(F * D, np.float32)
        e = np.zeros(F, np.float32)
        for r0 in range(0, F, 256):
            x = G(G(20))
            for r in range(r0, r0 + 256):
                t = x & 16383
                j = y[t]
                x = q[j] * (x >> 14) + t - u[j]
                k = j >> 2
                if k:
                    e[r] = j % 4 + m
                    M = B[k][D]
                    v = x % M
                    x //= M
                    g = x & (1 << k) - 1
                    x >>= k
                    c = D
                    for i in range(k, 0, -1):
                        c -= 1
                        b = B[i]
                        while b[c] > v:
                            c -= 1
                        v -= b[c]
                        w[r * D + c] = 1 - 2 * (g >> i - 1 & 1)
        _W = w.reshape(F, D) * s * np.exp2(e / 2)[:, None]
    return _W


def embed(docs):
    hw = HashingVectorizer(n_features=FW, ngram_range=(1, 2), analyzer="word", alternate_sign=False, norm=None, dtype=np.float32)
    hc = HashingVectorizer(n_features=FC, ngram_range=(3, 5), analyzer="char_wb", alternate_sign=False, norm=None, dtype=np.float32)
    X = vstack([hstack([hw.transform(docs[i:i + 4000]), hc.transform(docs[i:i + 4000])]).tocsr() for i in range(0, len(docs), 4000)]).tocsr()
    X.data = np.log1p(X.data)
    if IDF:  # batch IDF, applied in place
        df = np.bincount(X.indices, minlength=X.shape[1])
        X.data *= (np.log((X.shape[0] + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)[X.indices]
    X = normalize(X, copy=False)
    Z = np.vstack([np.asarray(X[i:i + 4000] @ table(), np.float32) for i in range(0, len(docs), 4000)])
    z = np.abs(Z).sum(1) < 1e-9
    if z.any():
        Z[z] = np.random.RandomState(0).normal(size=(int(z.sum()), D)) * 1e-3
    return normalize(Z)


def smooth(Z, k, alpha, iters):
    ix = NearestNeighbors(n_neighbors=min(k + 1, len(Z)), metric="cosine").fit(Z).kneighbors(Z)[1][:, 1:]
    for _ in range(iters):
        Z = normalize((1 - alpha) * Z + alpha * Z[ix].mean(1))
    return Z


def langs(lab, docs):
    """Split, in place, the clusters of `lab` that hold several languages (social posts only).

    Insurance against a language community the table has never seen: the table then puts all those posts into one
    cluster, while the reference keeps the large languages apart.  Every cluster of at least 40 texts is regrouped by
    the batch's own words; groups of at least 25 texts leave the cluster unless their character profiles say they are
    the same or a related language.  A cluster in one language (nearly all of them) comes out as one group and stays.

    Never raises: the whole loop body is guarded.  The numbers (40, 4, 24, k 15 / alpha 0.4 / 4 iterations, cut 0.6,
    1/8, 25, 0.13) were checked as a set on rounds 61-74.  Keep the merge cut 0.13 (cosine 0.87) within 0.11-0.16
    (cosine 0.84-0.89): with a table trained up to the previous round the gain is gone at 0.10 and negative below it.
    """
    for j in np.flatnonzero(np.bincount(lab) > 39):
        F = np.flatnonzero(lab == j)
        T = [docs[i] for i in F]
        try:  # anything that fails (too few shared words, fewer than two groups) leaves the cluster as it is
            # words of 2+ letters found in 4+ texts -> 24-d LSA -> k-NN smoothing -> average linkage cut at cosine distance 0.6
            U = TruncatedSVD(24, random_state=0).fit_transform(TfidfVectorizer(binary=True, use_idf=False, min_df=4, token_pattern=r"\b[^\W\d_]{2,}\b").fit_transform(T))
            z = abs(U).sum(1) < 1e-9
            U[z] = np.random.RandomState(0).randn(z.sum(), 24)
            g = fcluster(linkage(smooth(normalize(U), 15, .4, 4), "average", "cosine"), .6, "distance")
            # script id 1-6 of a text when at least 1/8 of its characters are Cyrillic (U+0400-052F) / CJK, Kana or Hangul
            # (U+4E00-9FFF, 3040-30FF, AC00-D7AF) / Arabic (0600-06FF) / Hebrew (0590-05FF) / Thai (0E00-0E7F) / Devanagari
            # (0900-097F), else 0 (abs: also 0 for an empty text); all texts of one such script form one group
            s = np.array([np.argmax([abs(len(d) / 8 - .1)] + [len(re.findall(p, d)) for p in ("[Ѐ-ԯ]", "[一-鿿぀-\u30ff가-힯]", "[؀-ۿ]", "[֐-׿]", "[฀-๿]", "[ऀ-ॿ]")]) for d in T])
            g = np.where(s, 9999 + s, g)
            u, c = np.unique(g, return_counts=True)
            u = u[c > 24]
            # groups whose character 1-3-gram profiles have cosine >= 0.87 (same or related language) are merged again
            C = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), use_idf=False, sublinear_tf=True, min_df=2).fit_transform(T)
            m = fcluster(linkage(np.vstack([np.asarray(C[g == b].mean(0)) for b in u]), "single", "cosine"), .13, "distance")
            G = sum((g == b) * x for b, x in zip(u, m))
            G[G == np.bincount(G)[1:].argmax() + 1] = 0  # the largest group keeps the old id, together with the leftover texts
            lab[F] = np.where(G, lab.max() + G, j)
        except Exception:
            continue


def core(Z, p, docs=None, g=None):
    k, alpha, iters, S, fr = p[:5]
    n = len(Z)
    Zs = smooth(Z, k, alpha, iters)
    nd, ni = NearestNeighbors(n_neighbors=min(31, n), metric="cosine").fit(Zs).kneighbors(Zs)
    d = nd[:, min(15, n - 1)]  # density: cosine distance to the 15th neighbour
    link = linkage(Zs, "average", "cosine")
    out = np.argsort(-d)[: int(n * fr)]
    if docs is not None and REFINE and n >= 500:
        try:  # self-training: a linear classifier on the batch's exact vocabulary learns the first-pass clusters
            pseudo = fcluster(link, REFINE[0], "maxclust")
            ok = np.ones(n, bool)
            ok[out] = False
            ok &= np.bincount(pseudo)[pseudo] >= 10
            kw = dict(max_features=60000, min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
            X = normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(docs),
                                  TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(docs)]).tocsr())
            P = RidgeClassifier(alpha=REFINE[1]).fit(X[ok], pseudo[ok]).decision_function(X)
            P = normalize(P - P.mean(1, keepdims=True))
            # second pass: no docs (no further self-training); the texts go along for the language split, social posts only
            return core(normalize(np.hstack([Z, REFINE[2] * P])), p, None, (d, P, np.log1p(X.getnnz(1)), LANG and p is P_SOCIAL and docs))
        except Exception:
            docs = None
    lab = fcluster(link, max(2, min(S, n // 8)), "maxclust").astype(np.int64)
    if g and g[3]:  # only in the second (self-trained) pass of a batch of social posts: g[3] = its texts
        langs(lab, g[3])
    if g:
        # Which points to abstain on.  `out` (the lowest-density points) is the starting choice; a linear ranker, fitted
        # offline on past rounds to the change of the batch score when one point becomes a singleton, re-picks the same
        # number of points.  Any failure (or a settings tuple without weights) keeps the density choice.
        try:
            d1, P, z = g[:3]  # first pass: density and classifier scores; log(1 + number of TF-IDF features) of each text
            I = np.arange(n)
            kp = ~np.isin(I, out)  # points the density rule keeps
            A = np.eye(lab.max() + 1)[lab] * kp[:, None]  # one-hot cluster of the kept points
            sk = A.sum(0)
            m = sk[lab]  # kept points in the point's cluster
            c = A.T @ Zs / np.maximum(sk, 1)[:, None]  # mean vector of each cluster's kept points
            s = Zs @ normalize(c).T  # cosine to every cluster centroid
            s[:, sk < 1] = -1
            o = s[I, lab]
            s[I, lab] = -1
            ld = np.log(d + 1e-6)
            dm = np.array([np.median(d[a > 0]) if a.any() else 1 for a in A.T])  # median density of each cluster's kept points
            q = np.sort(P, 1)
            r = np.argsort(np.argsort([d, d1])) / n
            nb = ni[:, 1:16]
            F = np.c_[
                r[0],  # rank of the density
                ld - np.median(ld),  # log density relative to the batch
                r[1],  # rank of the first-pass density
                np.log(nd[:, [5, 30]] + 1e-6) - ld[:, None],  # log distance to the 5th and 30th neighbour relative to the 15th
                q[:, -1],  # top classifier score
                q[:, -1] - q[:, -2],  # its margin over the second
                np.log(m + ~kp),  # log size of the cluster (its kept points, plus the point itself if it is not one)
                (lab[nb] == lab[:, None]).mean(1),  # share of the 15 neighbours in the same cluster
                kp[nb].mean(1),  # share of the 15 neighbours that are kept
                o,  # cosine to the own centroid
                o - s.max(1),  # ... minus the best other centroid
                1 - m / np.bincount(lab)[lab],  # share of the cluster already abstained
                ld - np.log(dm[lab] + 1e-6),  # log density relative to the cluster
                (c * c).sum(1)[lab],  # squared norm of the cluster's mean vector (cohesion)
                (A.T @ (P == q[:, -1:])).argmax(1)[lab] == P.argmax(1),  # top class = the cluster's most common top class
                z - np.median(z)  # text length relative to the batch
            ] @ p[5]
            F[m < 1] = 9e9  # a cluster with no kept point stays abstained
            out = np.argsort(-F)[: len(out)]
        except Exception:
            g = 0
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
            lab = core(Z, p, sub)
        else:  # cluster a sample, give the rest the majority label of their 5 nearest sampled points
            pick = np.sort(np.random.default_rng(0).choice(m, MAXN, replace=False))
            base = core(Z[pick], p)
            ix = NearestNeighbors(n_neighbors=5, metric="cosine").fit(Z[pick]).kneighbors(Z)[1]
            lab = np.array([np.bincount(r).argmax() for r in np.unique(base, return_inverse=True)[1][ix]])
    except Exception:
        lab = fallback(sub)
    lab = np.unique(lab, return_inverse=True)[1]
    if LINK:  # Reddit image posts with a short caption: the reference embeds the raw text, where the link dominates
        w = [len(d.split()) for d in sub]
        one = np.bincount(lab)[lab] < 2
        g = [i for i, t in enumerate(raw) if "preview.redd.it" in str(t) and (w[i] < 4 or w[i] < 11 and one[i])]
        if len(g) >= LINK:
            lab[g] = lab.max() + 1
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
