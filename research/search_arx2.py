import sys, numpy as np; sys.path.insert(0, "research")
import search as S
from concurrent.futures import ProcessPoolExecutor
def job(a): i, p = a; return S.run(S.D[i], p)
if __name__ == "__main__":
    S.D = S.load(); idx = [i for i, c in enumerate(S.D) if c["arx"]]; rnd = np.array([S.D[i]["rnd"] for i in idx]); b = S.V4[True]
    C = {"v4": b, "A S80 fr.3": {**b, "S": 80, "fr": .3}, "B A+k2 1.5": {**b, "S": 80, "fr": .3, "k2": 1.5}, "C B+Sp60": {**b, "S": 80, "fr": .3, "k2": 1.5, "Sp": 60},
         "D C+fr1 .25": {**b, "S": 80, "fr": .3, "k2": 1.5, "Sp": 60, "fr1": .25}, "E D+a.3": {**b, "S": 80, "fr": .3, "k2": 1.5, "Sp": 60, "fr1": .25, "a": .3},
         "F D+k20": {**b, "S": 80, "fr": .3, "k2": 1.5, "Sp": 60, "fr1": .25, "k": 20}, "G D S100": {**b, "S": 100, "fr": .3, "k2": 1.5, "Sp": 60, "fr1": .25},
         "H D S60": {**b, "S": 60, "fr": .3, "k2": 1.5, "Sp": 60, "fr1": .25}, "I D fr.35": {**b, "S": 80, "fr": .35, "k2": 1.5, "Sp": 60, "fr1": .25}, "J D fr.25": {**b, "S": 80, "fr": .25, "k2": 1.5, "Sp": 60, "fr1": .25},
         "K only S80": {**b, "S": 80}, "L only S120": {**b, "S": 120}}
    names = list(C)
    with ProcessPoolExecutor(14) as ex: R = np.array(list(ex.map(job, [(i, C[n]) for n in names for i in idx]))).reshape(len(names), len(idx))
    for n, r in zip(names, R): print("%-14s all %.4f (%+.4f)  odd %+.4f even %+.4f  rounds 69-74 %+.4f  wins %d/14" % (n, r.mean(), r.mean() - R[0].mean(), r[rnd % 2 == 1].mean() - R[0][rnd % 2 == 1].mean(), r[rnd % 2 == 0].mean() - R[0][rnd % 2 == 0].mean(), r[rnd >= 69].mean() - R[0][rnd >= 69].mean(), (r > R[0]).sum()))
