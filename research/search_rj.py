import sys, numpy as np; sys.path.insert(0, "research")
import search as S
from concurrent.futures import ProcessPoolExecutor
def job(a): i, p = a; return S.run(S.D[i], p)
if __name__ == "__main__":
    S.D = S.load(); arx = np.array([c["arx"] for c in S.D]); rnd = np.array([c["rnd"] for c in S.D])
    def P(i, **kw): b = dict(S.V4[bool(arx[i])]); b.update(dict(S=80, fr=.3) if arx[i] else {}); b.update({k: (v[int(arx[i])] if isinstance(v, tuple) else v) for k, v in kw.items()}); return b
    V = {"v4.1": {}, "rejoin 0.97": dict(rj=.97), "rejoin 0.95": dict(rj=.95), "rejoin 0.93": dict(rj=.93), "rejoin 0.90": dict(rj=.9), "rejoin .95 + share +.03": dict(rj=.95, fr=(.23, .33)), "rejoin .93 + share +.05": dict(rj=.93, fr=(.25, .35))}
    names = list(V)
    with ProcessPoolExecutor(14) as ex: R = np.array(list(ex.map(job, [(i, P(i, **V[n])) for n in names for i in range(len(S.D))]))).reshape(len(names), len(S.D))
    rs = lambda r, m: (3 * r[m & ~arx].mean() + r[m & arx].mean()) / 4
    for n, r in zip(names, R): print("%-26s round-level all %.4f (%+.4f) | odd %+.4f even %+.4f | social %+.4f arxiv %+.4f | subset wins %d/%d" % (n, rs(r, rnd > 0), rs(r, rnd > 0) - rs(R[0], rnd > 0), rs(r, rnd % 2 == 1) - rs(R[0], rnd % 2 == 1), rs(r, rnd % 2 == 0) - rs(R[0], rnd % 2 == 0), r[~arx].mean() - R[0][~arx].mean(), r[arx].mean() - R[0][arx].mean(), (r > R[0] + 1e-9).sum(), len(r)))
