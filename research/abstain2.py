import pickle, numpy as np, sys, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi, roc_auc_score
sc = lambda g, p: (max(0, ari(g, p)) + nmi(g, p)) / 2
R = pickle.load(open("cache/abstain_69_73.pkl", "rb")); names = R[0][5]; print(names)
rk = lambda F: np.argsort(np.argsort(F, 0), 0) / len(F)
rnd = np.repeat(np.arange(5), 4)   # files are ordered by round, 4 per round
def run(model_fn, target, label, use=None):
    out = {False: [], True: []}; aucs = []
    for r in range(5):
        tr = [i for i in range(20) if rnd[i] != r]; te = [i for i in range(20) if rnd[i] == r]
        cols = [names.index(c) for c in use] if use else list(range(len(names)))
        Xtr = np.vstack([np.column_stack([rk(R[i][2])[:, cols], np.full(len(R[i][3]), float(R[i][0]))]) for i in tr]); ytr = np.concatenate([R[i][target] for i in tr])
        m = model_fn().fit(Xtr, ytr)
        for i in te:
            arx, _, F, bad, y, _, l0, gt = R[i]; p = m.predict_proba(np.column_stack([rk(F)[:, cols], np.full(len(gt), float(arx))]))[:, 1]; aucs.append(roc_auc_score(R[i][target], p))
            for fr in (.2, .3, .4):
                o = np.argsort(-p)[: int(len(gt) * fr)]; l = l0.copy(); l[o] = 10**6 + np.arange(len(o)); out[arx].append((fr, sc(gt, l)))
    best = {a: max(np.mean([s for f, s in out[a] if f == fr]) for fr in (.2, .3, .4)) for a in out}
    print("%-44s round %.4f social %.4f arxiv %.4f  AUC %.3f" % (label, (3 * best[False] + best[True]) / 4, best[False], best[True], np.mean(aucs)))
base = {False: [], True: []}
for arx, res, F, bad, y, _, l0, gt in R: base[arx].append(res["current"])
print("%-44s round %.4f" % ("current (knn distance)", (3 * np.mean(base[False]) + np.mean(base[True])) / 4))
run(lambda: LogisticRegression(max_iter=500), 3, "logistic -> bad, all signals")
run(lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=.1), 3, "GBM -> bad, all signals")
run(lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=.1), 4, "GBM -> noise, all signals")
run(lambda: LogisticRegression(max_iter=500), 3, "logistic -> bad, knn+knn30+maxp+small", ["knn", "knn30", "maxp", "small"])
