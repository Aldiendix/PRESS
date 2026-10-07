"""Unified test bench for PRESS experiments. Read docs/AGENT_BRIEF.md first.

Every table was trained ONLY on rounds before its test rounds, so all scores are on unseen data:
  D  rounds 61-74 (56 subsets), table trained on rounds 40-60   <- main set, lowest noise
  A  rounds 69-73 (20 subsets), table trained on rounds 40-68
  B  round 74      (4 subsets), table trained on rounds 40-73
  C  round 73      (4 subsets), table trained on rounds 40-72
A/B/C contain texts that are also in D, but with fresher tables (closer to the deployed condition).

Typical use (run with /root/PRESS/.venv-research/bin/python from /root/PRESS, OMP_NUM_THREADS=1):

    import sys; sys.path.insert(0, "research")
    import numpy as np, bench

    def my_variant(ctx):                      # ctx: one 5,000-text subset
        r = bench.pipeline(ctx, detail=True)   # the current v4.1 pipeline with all intermediates
        labels = r["labels"].copy()
        ...                                    # change something; NEVER read ctx["gt"] here
        return labels                          # or {"variant a": labels_a, "variant b": labels_b}

    if __name__ == "__main__":
        bench.compare(my_variant, sets=("D",), jobs=4)
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from sklearn.linear_model import RidgeClassifier  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

SETS = {
    "D": ("cache/W_w60.npy", list(range(61, 75))),
    "A": ("cache/W_w2.npy", [69, 70, 71, 72, 73]),
    "B": ("cache/W_nU73.npy", [74]),
    "C": ("cache/W_nU72.npy", [73]),
}
# v4.1 settings.  False = social posts, True = arXiv titles.
SETTINGS = {
    False: dict(k=20, alpha=0.4, iters=4, clusters=120, share=0.2),
    True: dict(k=35, alpha=0.4, iters=4, clusters=80, share=0.3),
}
REFINE = dict(classes=90, ridge_alpha=10.0, weight=0.75)
FW, FC = 8192, 16384


def score(gt, labels):
    """Official per-subset score: (max(0, ARI) + NMI) / 2 over all points (noise -1 is an ordinary label)."""
    return (max(0.0, adjusted_rand_score(gt, labels)) + normalized_mutual_info_score(gt, labels)) / 2


def smooth(Z, k, alpha=0.4, iters=4):
    """k-NN mean smoothing: Z <- normalize((1-alpha) Z + alpha * mean of the k nearest neighbours), `iters` times."""
    ix = NearestNeighbors(n_neighbors=min(k + 1, len(Z)), metric="cosine").fit(Z).kneighbors(Z)[1][:, 1:]
    for _ in range(iters):
        Z = normalize((1 - alpha) * Z + alpha * Z[ix].mean(1))
    return Z


def density(Zs, kth=15):
    """Cosine distance to the kth nearest neighbour (large = low density)."""
    return NearestNeighbors(n_neighbors=min(kth + 1, len(Zs)), metric="cosine").fit(Zs).kneighbors(Zs)[0][:, -1]


def pipeline(ctx, settings=None, refine=None, detail=False):
    """The v4.1 pipeline on one subset. Returns labels, or a dict with every intermediate when detail=True.

    Steps: student embedding Z -> smoothing -> average linkage -> self-training (ridge classifier on the exact
    TF-IDF matrix X learns the first-pass clusters of the dense points; its centred, normalised scores P are
    appended to Z) -> smoothing -> average linkage cut at `clusters` -> the `share` lowest-density points become
    singleton clusters (unique ids >= 10**6).
    """
    s = dict(SETTINGS[bool(ctx["arx"])])
    s.update(settings or {})
    r = dict(REFINE)
    r.update(refine or {})
    Z, X = ctx["Z"], ctx["X"]
    n = len(Z)
    Zs1 = smooth(Z, s["k"], s["alpha"], s["iters"])
    d1 = density(Zs1)
    link1 = linkage(Zs1, "average", "cosine")
    out1 = np.argsort(-d1)[: int(n * s["share"])]
    pseudo = fcluster(link1, r["classes"], "maxclust")
    ok = np.ones(n, bool)
    ok[out1] = False
    ok &= np.bincount(pseudo)[pseudo] >= 10
    P = RidgeClassifier(alpha=r["ridge_alpha"]).fit(X[ok], pseudo[ok]).decision_function(X)
    P = normalize(P - P.mean(1, keepdims=True))
    ZA = normalize(np.hstack([Z, r["weight"] * P])).astype(np.float32)
    Zs = smooth(ZA, s["k"], s["alpha"], s["iters"])
    d = density(Zs)
    link = linkage(Zs, "average", "cosine")
    clusters = fcluster(link, max(2, min(s["clusters"], n // 8)), "maxclust").astype(np.int64)
    out = np.argsort(-d)[: int(n * s["share"])]
    labels = clusters.copy()
    labels[out] = 10**6 + np.arange(len(out))
    if not detail:
        return labels
    return dict(labels=labels, clusters=clusters, singletons=out, Zs=Zs, d=d, link=link, ZA=ZA, P=P, pseudo=pseudo,
                train_mask=ok, Zs1=Zs1, d1=d1, link1=link1, settings=s, refine=r)


# ----------------------------------------------------------------------------------------------- data loading
def _files(rounds):
    import glob
    return [f for r in rounds for f in sorted(glob.glob(os.path.join(ROOT, f"data/round_{r:04d}/*.parquet")))]


def _exact_tfidf(texts):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    kw = dict(max_features=60000, min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
    return normalize(hstack([TfidfVectorizer(ngram_range=(1, 2), **kw).fit_transform(texts),
                             TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **kw).fit_transform(texts)]).tocsr())


def student_embedding(f, table, fw=FW, fc=FC):
    """L2-normalised student embedding of one round file under a table (numpy array, rows = fw + fc)."""
    from train import load_subset
    Z = np.asarray(load_subset(f, fw, fc, True)["X"] @ table, np.float32)
    zero = np.abs(Z).sum(1) < 1e-9
    if zero.any():
        Z[zero] = np.random.RandomState(0).normal(size=(int(zero.sum()), Z.shape[1])) * 1e-3
    return normalize(Z)


def texts(ctx, raw=False):
    """Texts of a subset: raw strings, or preprocessed exactly as the solution does (default)."""
    t = pd.read_parquet(ctx["f"], columns=["text"]).text.tolist()
    if raw:
        return t
    from feats import prep
    return [prep(x) for x in t]


def _ctx(args):
    f, table_path, fw, fc = args
    df = pd.read_parquet(f, columns=["label"])
    c = dict(f=f, rnd=int(f.split("round_")[1][:4]), arx=("arxiv" in f or "subset_4" in f), gt=df.label.values.astype(np.int64))
    c["Z"] = student_embedding(f, np.load(table_path), fw, fc)
    c["X"] = _exact_tfidf(texts(c))
    return c


def make_contexts(table_path, rounds, fw=FW, fc=FC, jobs=4):
    """Contexts for an arbitrary table (for example one you trained)."""
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(_ctx, [(f, table_path, fw, fc) for f in _files(rounds)]))


def load(name="D", jobs=4):
    """List of contexts for a bench set. Keys: f (parquet path), rnd, arx, gt, Z (student embedding), X (exact TF-IDF)."""
    p = os.path.join(ROOT, f"cache/bench_{name}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    table, rounds = SETS[name]
    D = make_contexts(os.path.join(ROOT, table), rounds, jobs=jobs)
    pickle.dump(D, open(p, "wb"))
    return D


# ----------------------------------------------------------------------------------------------- evaluation
_FN = None
_CTX = None


def _run(i):
    out = _FN(_CTX[i])
    if not isinstance(out, dict):
        out = {"variant": out}
    return {k: score(_CTX[i]["gt"], np.asarray(v)) for k, v in out.items()}


def _map(fn, ctxs, jobs):
    global _FN, _CTX
    _FN, _CTX = fn, ctxs
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(_run, range(len(ctxs))))


def baseline_scores(name="D", jobs=4):
    """Per-subset scores of the v4.1 pipeline on a set (cached)."""
    p = os.path.join(ROOT, f"cache/bench_base_{name}.npy")
    if os.path.exists(p):
        return np.load(p)
    b = np.array([r["variant"] for r in _map(pipeline, load(name), jobs)])
    np.save(p, b)
    return b


def compare(fn, sets=("D",), jobs=4, quiet=False):
    """Paired comparison of fn(ctx) -> labels (or {name: labels}) against the v4.1 baseline on the same subsets.

    Prints, per variant and set: mean score, delta vs baseline, the delta on odd and even rounds separately
    (split-half check), the delta on social and arXiv subsets, and how many subsets got better / worse.
    Returns {variant: {set: {...}}}.  A change is only credible if `delta`, `odd` and `even` are all positive on D.
    """
    res = {}
    for name in sets:
        C = load(name)
        base = baseline_scores(name, jobs)
        R = _map(fn, C, jobs)
        rnd = np.array([c["rnd"] for c in C])
        arx = np.array([bool(c["arx"]) for c in C])
        for k in R[0]:
            v = np.array([r[k] for r in R])
            dlt = v - base
            res.setdefault(k, {})[name] = dict(
                score=float(v.mean()), base=float(base.mean()), delta=float(dlt.mean()),
                odd=float(dlt[rnd % 2 == 1].mean()) if (rnd % 2 == 1).any() else 0.0,
                even=float(dlt[rnd % 2 == 0].mean()) if (rnd % 2 == 0).any() else 0.0,
                social=float(dlt[~arx].mean()), arxiv=float(dlt[arx].mean()),
                better=int((dlt > 1e-9).sum()), worse=int((dlt < -1e-9).sum()), n=len(v))
    if not quiet:
        for k, per in res.items():
            for name, r in per.items():
                print("%-44s set %s: %.4f (base %.4f) delta %+.4f | odd %+.4f even %+.4f | social %+.4f arxiv %+.4f | better %d worse %d of %d"
                      % (k[:44], name, r["score"], r["base"], r["delta"], r["odd"], r["even"], r["social"], r["arxiv"], r["better"], r["worse"], r["n"]), flush=True)
    return res


def table_score(table_path, fw=FW, fc=FC, jobs=4, min_round=61):
    """Round-level score on rounds 61-74 (set D texts) of a table that was trained ONLY on rounds <= 60, v4.1 pipeline."""
    table = np.load(table_path)
    base = [c for c in load("D") if c["rnd"] >= min_round]
    global _TAB
    _TAB = (table, fw, fc)
    ctxs = [dict(c) for c in base]
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:
        Z = list(ex.map(_embed_for_table, [c["f"] for c in ctxs]))
    for c, z in zip(ctxs, Z):
        c["Z"] = z
    v = np.array([r["variant"] for r in _map(pipeline, ctxs, jobs)])
    arx = np.array([bool(c["arx"]) for c in ctxs])
    rnd = np.array([c["rnd"] for c in ctxs])
    out = dict(score=float(v.mean()), odd=float(v[rnd % 2 == 1].mean()), even=float(v[rnd % 2 == 0].mean()),
               social=float(v[~arx].mean()), arxiv=float(v[arx].mean()), n=len(v))
    print("%s: round score %.4f (odd %.4f even %.4f) social %.4f arxiv %.4f on %d subsets"
          % (os.path.basename(table_path), out["score"], out["odd"], out["even"], out["social"], out["arxiv"], out["n"]), flush=True)
    return out


_TAB = None


def _embed_for_table(f):
    return student_embedding(f, *_TAB)


if __name__ == "__main__":
    # `python research/bench.py`            -> baseline on every set
    # `python research/bench.py table.npy`  -> score a table trained on rounds <= 60 (optionally: FW FC)
    if len(sys.argv) > 1:
        a = sys.argv[1:]
        table_score(a[0], *(int(x) for x in a[1:3])) if len(a) >= 3 else table_score(a[0])
    else:
        for name in SETS:
            b = baseline_scores(name)
            C = load(name)
            arx = np.array([bool(c["arx"]) for c in C])
            print("set %s: %d subsets, baseline %.4f (social %.4f, arxiv %.4f)" % (name, len(b), b.mean(), b[~arx].mean(), b[arx].mean()))
