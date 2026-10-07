"""Score a table trained on rounds <= 60 on unseen rounds 61-74 with the v4.1 pipeline. Usage: eval_D.py table.npy [FW FC]"""
import sys, numpy as np; sys.path.insert(0, "research")
import search as S
from concurrent.futures import ProcessPoolExecutor
WP = sys.argv[1]; FW, FC = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (8192, 16384)
def job(i):
    from train import load_subset; from grid2 import fixz
    c = dict(S.D[i]); c["Z"] = fixz(load_subset(c["f"], FW, FC, True)["X"] @ np.load(WP)).astype(np.float32)
    p = dict(S.V4[c["arx"]]); p.update(dict(S=80, fr=.3) if c["arx"] else {}); return S.run(c, p)
if __name__ == "__main__":
    S.D = S.load(); arx = np.array([c["arx"] for c in S.D]); rnd = np.array([c["rnd"] for c in S.D])
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 7) as ex: r = np.array(list(ex.map(job, range(len(S.D)))))
    import os as _o; keep = rnd >= int(_o.environ.get("MINR", "0")); r = np.where(keep, r, np.nan); f = lambda m: (3 * np.nanmean(r[m & ~arx]) + np.nanmean(r[m & arx])) / 4
    print("%s round %.4f (odd %.4f even %.4f) social %.4f arxiv %.4f" % (WP.split("/")[-1], f(rnd > 0), f(rnd % 2 == 1), f(rnd % 2 == 0), np.nanmean(r[~arx]), np.nanmean(r[arx])), flush=True)
