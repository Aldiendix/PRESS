"""Is ground-truth noise predictable from text alone across rounds? (train rounds 40-68, test 69-73)"""
import sys, numpy as np, scipy.sparse as sp, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
def ld(f): s = load_subset(f, 2**17, 2**17, True); return s["X"], (s["y"] == -1), s["arxiv"]
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: tr = list(ex.map(ld, files_for(range(40, 69)))); te = list(ex.map(ld, files_for(range(69, 74))))
    for grp in (False, True):
        Xtr = sp.vstack([x for x, y, a in tr if a == grp]); ytr = np.concatenate([y for x, y, a in tr if a == grp])
        clf = LogisticRegression(C=1.0, max_iter=200, solver="saga").fit(Xtr, ytr)
        aucs, p2, p5, p10, base = [], [], [], [], []
        for x, y, a in te:
            if a != grp: continue
            p = clf.decision_function(x); o = np.argsort(-p); aucs.append(roc_auc_score(y, p)); base.append(y.mean())
            p2.append(y[o[:100]].mean()); p5.append(y[o[:250]].mean()); p10.append(y[o[:500]].mean())
        print("arxiv" if grp else "social", "AUC %.3f  base %.2f  prec@2%% %.2f  @5%% %.2f  @10%% %.2f" % tuple(map(np.mean, (aucs, base, p2, p5, p10))), flush=True)
