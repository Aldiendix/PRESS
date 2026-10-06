"""Float probe: does a residual MLP on top of the hashed-bag embedding beat the linear student? (train 40-68, eval 71-73)"""
import sys, numpy as np, torch, torch.nn.functional as Fn, time, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from train import load_subset, files_for, to_torch
from evalw import quick_scores
H = int(sys.argv[1]); D = 96; FW, FC = 8192, 16384; torch.set_num_threads(6); torch.manual_seed(0); rng = np.random.default_rng(0)
S = [load_subset(f, FW, FC, True) for f in files_for(range(40, 69))]
W = torch.nn.Parameter(torch.randn(FW + FC, D) * 0.05); P = [W]
if H:
    A = torch.nn.Linear(D, H); B = torch.nn.Linear(H, D); torch.nn.init.zeros_(B.weight); P += list(A.parameters()) + list(B.parameters())
def emb(X):
    h = Fn.normalize(torch.sparse.mm(X, W), dim=1)
    return Fn.normalize(h + B(torch.relu(A(h * 4))), dim=1) if H else h
steps = 5000; opt = torch.optim.Adam(P, lr=0.02); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.02, total_steps=steps, pct_start=0.05); bs = 1536; eye = torch.eye(bs, dtype=torch.bool)
for step in range(steps):
    s = S[rng.integers(len(S))]; idx = rng.choice(5000, bs, replace=False); z = emb(to_torch(s["X"][idx])); y = torch.from_numpy(s["y"][idx])
    lg = (z @ z.T / 0.1).masked_fill(eye, -1e9); logp = lg - torch.logsumexp(lg, 1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & (y[:, None] >= 0) & ~eye; has = pos.any(1); loss = -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()
    x = torch.from_numpy(s["xyz"][idx]); xl = (-torch.cdist(x, x) ** 2 / 0.25).masked_fill(eye, -1e9); loss = loss + 0.5 * -(torch.softmax(xl, 1) * logp).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
R = []
with torch.no_grad():
    for f in files_for([71, 72, 73]):
        s = load_subset(f, FW, FC, True); r = quick_scores(emb(to_torch(s["X"])).numpy(), s["y"]); r["arxiv"] = s["arxiv"]; R.append(r)
for g in (False, True):
    sel = [r for r in R if r["arxiv"] == g]; print("H=%d" % H, "arxiv" if g else "social", "agg70 %.4f pur %.4f" % (np.mean([r["agg70"] for r in sel]), np.mean([r["knn_pur"] for r in sel])), flush=True)
