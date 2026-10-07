import sys, numpy as np, pickle; sys.path.insert(0, "research")
import search as S
from concurrent.futures import ProcessPoolExecutor
def job(a): i, p = a; return S.run(S.D[i], p)
if __name__ == "__main__":
    S.D = S.load(); idx = [i for i, c in enumerate(S.D) if c["arx"]]; rnd = np.array([S.D[i]["rnd"] for i in idx]); rng = np.random.default_rng(1)
    def smp(): return dict(k=int(rng.choice([12, 16, 20, 26, 35, 50])), a=float(rng.choice([.3, .4, .5, .6])), it=int(rng.choice([4, 5, 6, 8])), Sp=int(rng.choice([40, 60, 90, 130])), al=float(rng.choice([5., 10., 20.])),
                           w=float(rng.choice([.3, .4, .5, .6, .75])), S=int(rng.choice([80, 120, 180, 260])), fr=float(rng.choice([.25, .3, .35, .4])), dk=int(rng.choice([10, 15, 25])), k2=float(rng.choice([1.0, 1.5, 2.0])), fr1=float(rng.choice([.15, .25, .35])))
    cfgs = [S.V4[True]] + [smp() for _ in range(160)]
    with ProcessPoolExecutor(14) as ex: R = np.array(list(ex.map(job, [(i, p) for p in cfgs for i in idx]))).reshape(len(cfgs), len(idx))
    pickle.dump((cfgs, R, rnd), open("cache/search_arx.pkl", "wb"))
    odd, even = R[:, rnd % 2 == 1].mean(1), R[:, rnd % 2 == 0].mean(1); m = R.mean(1); wins = (R > R[0]).sum(1)
    print("v4: all %.4f (odd %.4f even %.4f)" % (m[0], odd[0], even[0]))
    for t in np.argsort(-np.minimum(odd - odd[0], even - even[0]))[:8]: print("  all %+.4f odd %+.4f even %+.4f  wins %d/14  %s" % (m[t] - m[0], odd[t] - odd[0], even[t] - even[0], wins[t], cfgs[t]))
    # marginal effect of each setting
    for key in ("k", "a", "it", "Sp", "al", "w", "S", "fr", "dk", "k2", "fr1"):
        vals = sorted(set(c[key] for c in cfgs[1:])); print("  %-4s" % key, "  ".join("%s:%+.4f" % (v, np.mean([m[i] for i, c in enumerate(cfgs) if i and c[key] == v]) - m[1:].mean()) for v in vals))
