"""Train a hashed n-gram -> embedding student from ground-truth cluster labels (+ teacher embeddings when cached).

Student: z = normalize( normalize(log1p(counts)) @ W ),  counts = [word 1-2gram hash (FW) | char_wb 3-5gram hash (FC)].
"""
import argparse, glob, os, time, numpy as np, pandas as pd, scipy.sparse as sp, torch, torch.nn.functional as Fn
from sklearn.preprocessing import normalize

def fold(M, F):
    M = M.tocoo(); return sp.csr_matrix((M.data, (M.row, M.col % F)), shape=(M.shape[0], F))

def load_subset(f, FW, FC, idf=False):
    b = os.environ.get("FEAT", "cache/feat") + "/" + os.path.basename(f).replace(".parquet", "")
    X = sp.hstack([fold(sp.load_npz(b + ".w.npz"), FW), fold(sp.load_npz(b + ".c.npz"), FC)]).tocsr() if FC else fold(sp.load_npz(b + ".w.npz"), FW)
    X.sum_duplicates(); X.data = np.log1p(X.data)
    if idf:
        df_ = np.asarray((X > 0).sum(0)).ravel(); X = X @ sp.diags(np.log((X.shape[0] + 1.0) / (df_ + 1.0)) + 1.0)
    X = normalize(X).astype(np.float32)
    df = pd.read_parquet(f, columns=["label", "x", "y", "z"])
    tp = "cache/" + os.path.basename(f).replace(".parquet", ".mpnet.npy")
    return dict(X=X, y=df.label.values.astype(np.int64), xyz=df[["x", "y", "z"]].values.astype(np.float32),
                T=np.load(tp) if os.path.exists(tp) else None, arxiv=("arxiv" in f or "subset_4" in f), name=os.path.basename(f))

def files_for(rounds):
    return [f for r in rounds for f in sorted(glob.glob(f"data/round_{r:04d}/*.parquet"))]

def to_torch(X):
    X = X.tocoo(); return torch.sparse_coo_tensor(np.vstack([X.row, X.col]), X.data, X.shape).coalesce()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", type=int, default=8192); ap.add_argument("--fc", type=int, default=4096); ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--train", type=int, nargs=2, default=[1, 68]); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=1536); ap.add_argument("--tau", type=float, default=0.1); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wt", type=float, default=1.0, help="teacher similarity loss weight"); ap.add_argument("--wx", type=float, default=0.0, help="xyz-neighbour loss weight")
    ap.add_argument("--tern", type=float, default=0.0, help="ternary QAT threshold (x mean|W| per dim); 0 = float")
    ap.add_argument("--noise_neg", type=int, default=1, help="use GT-noise points as negatives in the label loss"); ap.add_argument("--nz", type=float, default=0.0, help="target non-zero fraction (overrides --tern threshold)"); ap.add_argument("--idf", type=int, default=0); ap.add_argument("--rows", type=int, default=0, help="per-row scale levels (0=off)")
    ap.add_argument("--l1", type=float, default=0.0); ap.add_argument("--only", default="all"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sm", type=int, default=0, help="in-loop k-NN smoothing iterations"); ap.add_argument("--smk", type=int, default=6); ap.add_argument("--sma", type=float, default=0.4)
    ap.add_argument("--smw", type=float, default=0.5, help="weight of the un-smoothed loss when --sm is on"); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    fs = files_for(range(a.train[0], a.train[1] + 1)); S = [load_subset(f, a.fw, a.fc, a.idf) for f in fs]
    if a.only != "all": S = [s for s in S if s["arxiv"] == (a.only == "arxiv")]
    print(len(S), "subsets; with teacher:", sum(s["T"] is not None for s in S), flush=True)
    F = a.fw + a.fc; W = torch.nn.Parameter(torch.randn(F, a.d) * 0.05); scale = torch.nn.Parameter(torch.ones(a.d)); g = torch.nn.Parameter(torch.zeros(F)); opt = torch.optim.Adam([W, scale, g], lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    def quant(W):
        if a.nz:
            thr = torch.kthvalue(W.detach().abs(), int(W.shape[0] * (1 - a.nz)), dim=0, keepdim=True).values
        else:
            thr = a.tern * W.abs().mean(0, keepdim=True)
        q = torch.sign(W) * (W.abs() > thr)
        Wq = (W + (q * W.abs().mean(0, keepdim=True) - W).detach()) * scale
        if a.rows: Wq = Wq * rowq()[:, None]
        return Wq, q
    def rowq():   # per-row scale on a half-octave grid, straight-through
        k = torch.clamp(torch.round(2 * g), -(a.rows // 2), a.rows - a.rows // 2 - 1); return torch.exp2((g + (k / 2 - g).detach()))
    t0 = time.time(); ema = None
    for step in range(a.steps):
        s = S[rng.integers(len(S))]; idx = rng.choice(s["X"].shape[0], a.bs, replace=False)
        Wf = quant(W)[0] if (a.tern or a.nz) else W
        z = Fn.normalize(torch.sparse.mm(to_torch(s["X"][idx]), Wf), dim=1)
        y = torch.from_numpy(s["y"][idx]); eye = torch.eye(a.bs, dtype=torch.bool)
        def objective(z):
            sim = z @ z.T
            logits = (sim / a.tau).masked_fill(eye, -1e9); logp = logits - torch.logsumexp(logits, 1, keepdim=True)
            logp_all = logp
            if not a.noise_neg:
                lg2 = logits.masked_fill((y < 0)[None, :], -1e9); logp = lg2 - torch.logsumexp(lg2, 1, keepdim=True)
            pos = (y[:, None] == y[None, :]) & (y[:, None] >= 0) & ~eye; has = pos.any(1)
            loss = -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()
            if a.wt and s["T"] is not None:   # match teacher neighbour distribution
                t = Fn.normalize(torch.from_numpy(s["T"][idx]), dim=1); tl = (t @ t.T / 0.05).masked_fill(eye, -1e9)
                loss = loss + a.wt * -(torch.softmax(tl, 1) * logp_all).sum(1).mean()
            if a.wx:
                x = torch.from_numpy(s["xyz"][idx]); d2 = torch.cdist(x, x) ** 2; xl = (-d2 / 0.25).masked_fill(eye, -1e9)
                loss = loss + a.wx * -(torch.softmax(xl, 1) * logp_all).sum(1).mean()
            return loss
        loss = objective(z)
        if a.sm:   # the pipeline clusters k-NN-smoothed vectors: train through the same smoothing
            nb = (z @ z.T).detach().masked_fill(eye, -1e9).topk(a.smk, dim=1).indices; zs = z
            for _ in range(a.sm): zs = Fn.normalize((1 - a.sma) * zs + a.sma * zs[nb].mean(1), dim=1)
            loss = a.smw * loss + objective(zs)
        if a.l1: loss = loss + a.l1 * W.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 500 == 0 or step == a.steps - 1: print(step, "loss %.4f" % ema, "%.0fs" % (time.time() - t0), flush=True)
    if a.tern or a.nz:
        q = quant(W)[1].detach().numpy().astype(np.int8); sc_ = scale.detach().numpy().astype(np.float32)
        np.save(a.out.replace(".npy", ".q.npy"), q); np.save(a.out.replace(".npy", ".s.npy"), sc_)
        nz = float((q != 0).mean()); p = np.array([1 - nz, nz / 2, nz / 2]); H = -(p * np.log2(p + 1e-12)).sum()
        print("nonzero %.3f entropy %.3f bits/w -> %.1f KB ideal (+%.1f KB row scales)" % (nz, H, H * q.size / 8 / 1024, F * np.log2(max(a.rows, 1)) / 8 / 1024), flush=True)
        Wout = q.astype(np.float32) * sc_
        if a.rows:
            rq = rowq().detach().numpy().astype(np.float32); np.save(a.out.replace(".npy", ".r.npy"), rq); Wout = Wout * rq[:, None]
            print("row levels used:", np.unique(np.round(2 * np.log2(rq)).astype(int), return_counts=True), flush=True)
        np.save(a.out, Wout)
    else:
        np.save(a.out, W.detach().numpy().astype(np.float32))
