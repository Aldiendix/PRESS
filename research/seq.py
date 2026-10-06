"""Model-class probe: hashed word sequence + L self-attention layers (+ char n-gram bag) vs pure bag (L=0).
Float weights, same supervised-contrastive loss; train rounds 40-68, report neighbour purity on unseen 71-73.
Usage: seq.py L [steps]"""
import sys, os, re, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, warnings, pickle
warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from sklearn.utils import murmurhash3_32
from train import load_subset, files_for, to_torch
from feats import prep
from evalw import quick_scores
L = int(sys.argv[1]); STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5000; D = 96; V = 32768; T = 64; FC = 16384
torch.set_num_threads(int(os.environ.get("NT", "5"))); torch.manual_seed(0); rng = np.random.default_rng(0)
TOK = re.compile(r"(?u)\b\w\w+\b")
def seqs(f):
    c = "cache/seq/" + os.path.basename(f).replace(".parquet", ".npy")
    if os.path.exists(c): return np.load(c)
    out = np.zeros((5000, T), np.int32)
    for i, t in enumerate(pd.read_parquet(f, columns=["text"]).text):
        w = TOK.findall(prep(t))[:T]
        if w: out[i, :len(w)] = [murmurhash3_32(x, positive=True) % (V - 1) + 1 for x in w]
    os.makedirs("cache/seq", exist_ok=True); np.save(c, out); return out
def load(f):
    s = load_subset(f, 8, FC, True); X = s["X"][:, 8:]   # char n-gram block only
    return dict(ids=seqs(f), Xc=X.tocsr(), y=s["y"], xyz=s["xyz"], arxiv=s["arxiv"])
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.E = nn.Embedding(V, D, padding_idx=0); s.C = nn.Parameter(torch.randn(FC, D) * .05); s.pos = nn.Parameter(torch.zeros(T, D)); nn.init.normal_(s.E.weight, std=.05)
        s.layers = nn.ModuleList([nn.TransformerEncoderLayer(D, 4, 192, dropout=0.1, batch_first=True, norm_first=True) for _ in range(L)]); s.att = nn.Linear(D, 1)
    def forward(s, ids, Xc):
        m = ids > 0; h = s.E(ids)
        if L:
            h = h + s.pos[None]
            mm = m.clone(); mm[:, 0] |= ~m.any(1)   # keep one position for empty texts
            for l in s.layers: h = l(h, src_key_padding_mask=~mm)
            w = torch.softmax(s.att(h).squeeze(-1).masked_fill(~mm, -1e9), 1) * m.any(1, keepdim=True); pooled = torch.nan_to_num((h * w[..., None]).sum(1))
        else:
            pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1).sqrt()
        return Fn.normalize(Fn.normalize(pooled, dim=1) + torch.sparse.mm(Xc, s.C), dim=1)
if __name__ == "__main__":
    S = [load(f) for f in files_for(range(40, 69))]; net = Net(); opt = torch.optim.Adam(net.parameters(), lr=0.02 if L == 0 else 0.004)
    for g in opt.param_groups: pass
    opt = torch.optim.Adam([{"params": [net.E.weight, net.C], "lr": 0.02}, {"params": [p for n_, p in net.named_parameters() if n_ not in ("E.weight", "C")], "lr": 0.002}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[0.02, 0.002], total_steps=STEPS, pct_start=0.05); bs = 1024; eye = torch.eye(bs, dtype=torch.bool); t0 = time.time()
    for step in range(STEPS):
        s = S[rng.integers(len(S))]; idx = np.sort(rng.choice(5000, bs, replace=False)); net.train()
        z = net(torch.from_numpy(s["ids"][idx]).long(), to_torch(s["Xc"][idx])); y = torch.from_numpy(s["y"][idx])
        lg = (z @ z.T / 0.1).masked_fill(eye, -1e9); logp = lg - torch.logsumexp(lg, 1, keepdim=True)
        pos = (y[:, None] == y[None, :]) & (y[:, None] >= 0) & ~eye; has = pos.any(1); loss = -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()
        x = torch.from_numpy(s["xyz"][idx]); xl = (-torch.cdist(x, x) ** 2 / 0.25).masked_fill(eye, -1e9); loss = loss + 0.5 * -(torch.softmax(xl, 1) * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0: print(step, "%.3f" % loss.item(), "%.0fs" % (time.time() - t0), flush=True)
    net.eval()
    for name, rounds in (("train 66-68", [66, 67, 68]), ("unseen 71-73", [71, 72, 73])):
        R = []
        with torch.no_grad():
            for f in files_for(rounds):
                s = load(f); z = torch.cat([net(torch.from_numpy(s["ids"][i:i + 1000]).long(), to_torch(s["Xc"][i:i + 1000])) for i in range(0, 5000, 1000)]).numpy()
                r = quick_scores(z, s["y"]); r["arxiv"] = s["arxiv"]; R.append(r)
        for g in (False, True):
            sel = [r for r in R if r["arxiv"] == g]; print("L=%d %s %s agg70 %.4f purity %.4f" % (L, name, "arxiv" if g else "social", np.mean([r["agg70"] for r in sel]), np.mean([r["knn_pur"] for r in sel])), flush=True)
