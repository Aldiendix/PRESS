"""Local replica of the Apex text-clustering scorer.

Usage: python tools/evaluate.py SOLUTION.py [--rounds 71 72 73] [--jobs 8] [--tag NAME]
Score per subset = (max(0, ARI) + NMI) / 2 against the `label` column (noise -1 is a normal label),
round score = mean over subsets -- verified to match official per-subset numbers exactly.
"""
import argparse, glob, importlib.util, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def run_one(args):
    sol, path = args
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    import numpy as np, pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    spec = importlib.util.spec_from_file_location("solution_under_test", sol)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = pd.read_parquet(path)
    t = time.time()
    pred = np.asarray(mod.cluster_texts(df.text.tolist()))
    dt = time.time() - t
    gt = df.label.values
    ari, nmi = adjusted_rand_score(gt, pred), normalized_mutual_info_score(gt, pred)
    ids, cnt = np.unique(pred, return_counts=True)
    return dict(file=os.path.basename(path), ari=ari, nmi=nmi, score=(max(0.0, ari) + nmi) / 2, sec=dt,
                n_pred=int(len(ids)), n_single=int((cnt == 1).sum()), n_gt=int(len(set(gt)) - 1), gt_noise=int((gt == -1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("--rounds", type=int, nargs="*")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tag")
    ap.add_argument("-q", action="store_true")
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(ROOT, "data", "round_*")))
    if a.rounds:
        dirs = [d for d in dirs if int(d[-4:]) in a.rounds]
    files = [f for d in dirs for f in sorted(glob.glob(os.path.join(d, "*.parquet")))]
    sol = os.path.abspath(a.solution)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run_one, [(sol, f) for f in files]))
    by_round = {}
    for r in res:
        by_round.setdefault(r["file"][:10], []).append(r)
        if not a.q:
            print(f"{r['file'][:28]:28s} score {r['score']:.4f} ari {r['ari']:.4f} nmi {r['nmi']:.4f} {r['sec']:5.1f}s "
                  f"pred {r['n_pred']:4d} (single {r['n_single']:4d}) gt {r['n_gt']:3d} noise {r['gt_noise']}")
    import numpy as np
    rs = {k: float(np.mean([x["score"] for x in v])) for k, v in by_round.items()}
    soc = float(np.mean([x["score"] for x in res if "arxiv" not in x["file"]]))
    arx = [x["score"] for x in res if "arxiv" in x["file"]]
    print(" ".join(f"{k[6:]}:{v:.4f}" for k, v in rs.items()))
    print(f"MEAN {np.mean(list(rs.values())):.4f}  social {soc:.4f}  arxiv {np.mean(arx) if arx else float('nan'):.4f}  "
          f"max_sec {max(x['sec'] for x in res):.1f}  rounds {len(rs)}")
    if a.tag:
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        json.dump(dict(solution=a.solution, rounds=rs, mean=float(np.mean(list(rs.values()))), subsets=res),
                  open(os.path.join(ROOT, "results", a.tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
