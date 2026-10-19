"""
lcl_core.py: everything one training run needs, in one place.

Same generator, models, training protocol, probe and Metric C as the pre-flight notebook
(all of which passed the pre-flight checks), with three additions:
  * sparsity of A (for the sparsity sweep),
  * depth / width variants of the plain MLP and mean-pool (for the depth sweep),
  * a memory-light probe (same numbers as before; needed above 16 features).
"""
import os, json, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = "lcl-cluster-v1"
QUICK = os.environ.get("LCL_QUICK", "0") == "1"        # tiny sizes, only to check the code runs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = False

SLATE, A_SCALE = 5, 1.0
N_VAL = N_EVAL = N_SHIFT = 300 if QUICK else 5000
MAX_EPOCHS = 15 if QUICK else 500
PATIENCE = 5 if QUICK else 60
CHUNK = 50000                     # sessions per gradient chunk; chunks add up to the exact full-batch gradient
PROBE_K_PER_D = 2 if QUICK else 10
PROBE_N = 60 if QUICK else 500
C_TRIALS = 300 if QUICK else 4000
HP_DEFAULT = dict(lr=1e-2, wd=0.0)


def n_boot(d):
    return 10 if QUICK else (200 if d <= 24 else 100)


# ============================================================ data
def make_ground_truth(d, rank, seed, sparsity=0.0, A_scale=A_SCALE):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=d); theta /= np.linalg.norm(theta)
    U, V = rng.normal(size=(d, rank)), rng.normal(size=(d, rank))
    A = U @ V.T
    if sparsity > 0:
        mask = np.ones(d * d)
        mask[rng.choice(d * d, int(round(sparsity * d * d)), replace=False)] = 0.0
        A = A * mask.reshape(d, d)
    return theta, A_scale * A / np.linalg.norm(A)


def simulate_choices(theta, A, X, seed):
    rng = np.random.default_rng(seed)
    util = np.einsum("nd,nsd->ns", theta[None, :] + X.mean(axis=1) @ A.T, X)
    z = util - util.max(axis=1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
    choices = np.clip((rng.random((len(X), 1)) > probs.cumsum(axis=1)).sum(axis=1), 0, X.shape[1] - 1)
    return probs, choices


def build_splits(theta, A, n_train, d, data_seed):
    rng = np.random.default_rng(data_seed)
    X_all = rng.normal(size=(n_train + N_VAL + N_EVAL, SLATE, d))
    pool = np.random.default_rng(data_seed + 1).normal(size=(N_SHIFT * 20, SLATE, d))
    X_shift = pool[np.argsort(np.linalg.norm(pool.mean(axis=1), axis=1))[-N_SHIFT:]]
    parts = {"train": X_all[:n_train], "val": X_all[n_train:n_train + N_VAL],
             "eval_in": X_all[n_train + N_VAL:], "eval_shift": X_shift}
    out = {}
    for k, (name, Xp) in enumerate(parts.items()):
        probs, choices = simulate_choices(theta, A, Xp, seed=data_seed * 10 + k)
        out[name] = dict(X=torch.tensor(Xp, dtype=torch.float32, device=DEVICE),
                         y=torch.tensor(choices, dtype=torch.long, device=DEVICE),
                         p=torch.tensor(probs, dtype=torch.float32, device=DEVICE))
    return out


# ============================================================ models
class MeanPoolNetwork(nn.Module):
    def __init__(self, d, hidden, n_layers=2):
        super().__init__()
        layers, inp = [], 2 * d
        for _ in range(n_layers):
            layers += [nn.Linear(inp, hidden), nn.ReLU()]; inp = hidden
        layers.append(nn.Linear(inp, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, X):
        y = X.mean(dim=1, keepdim=True).expand(-1, X.shape[1], -1)
        return self.net(torch.cat([X, y], dim=-1)).squeeze(-1)


class SelfAttentionModel(nn.Module):
    def __init__(self, d, d_model=32, n_heads=2, n_layers=1):
        super().__init__()
        self.embed = nn.Linear(d, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)
    def forward(self, X):
        return self.head(self.encoder(self.embed(X))).squeeze(-1)


class FlatMLP(nn.Module):
    def __init__(self, d, n_layers, width):
        super().__init__()
        layers, inp = [], SLATE * d
        for _ in range(n_layers):
            layers += [nn.Linear(inp, width), nn.ReLU()]; inp = width
        layers.append(nn.Linear(inp, SLATE))
        self.net = nn.Sequential(*layers)
    def forward(self, X):
        return self.net(X.reshape(X.shape[0], -1))


class BilinearAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.Wq = nn.Parameter(torch.randn(h, d) * 0.1)
        self.Wk = nn.Parameter(torch.randn(h, d) * 0.1)
    def forward(self, X):
        q, k = X @ self.Wq.T, X @ self.Wk.T
        return X @ self.w + torch.einsum("nih,njh->ni", q, k) / X.shape[1]


class SoftmaxOV(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.Wq, self.Wk = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False)
        self.Wv, self.Wo = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False)
        for lin in (self.Wq, self.Wk, self.Wv, self.Wo): nn.init.normal_(lin.weight, std=0.1)
        self.h = h
    def forward(self, X):
        att = torch.softmax(self.Wq(X) @ self.Wk(X).transpose(1, 2) / math.sqrt(self.h), dim=-1)
        return X @ self.w + (self.Wo(X) * (att @ self.Wv(X))).sum(-1)


class LCLOracle(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(d) * 0.1)
        self.A = nn.Parameter(torch.randn(d, d) * 0.1)
    def forward(self, X):
        return torch.einsum("nd,nsd->ns", self.theta.unsqueeze(0) + X.mean(dim=1) @ self.A.T, X)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _closest_width(cost, target):
    return min(range(2, 3000), key=lambda w: abs(cost(w) - target))


_WIDTHS = {}
def base_width(kind, d, L):
    """Width that gives the same parameter count as 1-layer self-attention at this d."""
    key = (kind, d, L)
    if key not in _WIDTHS:
        target = n_params(SelfAttentionModel(d))
        if kind == "mp":
            cost = lambda w: (2 * d) * w + w + (L - 1) * (w * w + w) + w + 1
        else:  # flat
            cost = lambda w: (SLATE * d) * w + w + (L - 1) * (w * w + w) + w * SLATE + SLATE
        _WIDTHS[key] = _closest_width(cost, target)
    return _WIDTHS[key]


def parse_name(name):
    if "_x" in name:
        base, mult = name.split("_x"); return base, float(mult)
    return name, 1.0


def build_model(name, d):
    """Model names:
       sa_L{L}      self-attention, L layers (d_model 32, 2 heads, pre-LN)
       mp_L{L}      mean-pool network, L hidden layers (width matched to sa_L1 at L=2, kept for deeper)
       flat_L{L}    plain MLP on the whole slate, L hidden layers, SAME SIZE as sa_L1
       flatw_L{L}   plain MLP, L hidden layers, SAME WIDTH as flat_L2 (grows with depth)
       bil_h{h}     bilinear attention (exactly LCL with rank <= h)
       sov_h{h}     one softmax head with feature read-out (= LCL when weights are uniform)
       oracle       the LCL formula with free theta and A
       suffix _x{m} multiplies the width / d_model by m (width robustness check)"""
    base, mult = parse_name(name)
    if base.startswith("sa_L"):
        return SelfAttentionModel(d, d_model=int(round(32 * mult)), n_layers=int(base[4:]))
    if base.startswith("mp_L"):
        L = int(base[4:]); return MeanPoolNetwork(d, int(round(base_width("mp", d, 2) * mult)), L)
    if base.startswith("flatw_L"):
        L = int(base[7:]); return FlatMLP(d, L, int(round(base_width("flat", d, 2) * mult)))
    if base.startswith("flat_L"):
        L = int(base[6:]); return FlatMLP(d, L, int(round(base_width("flat", d, L) * mult)))
    if base.startswith("bil_h"):
        return BilinearAttention(d, int(base[5:]))
    if base.startswith("sov_h"):
        return SoftmaxOV(d, int(base[5:]))
    if base == "oracle":
        return LCLOracle(d)
    raise ValueError(name)


def family(name):
    base, _ = parse_name(name)
    if base.startswith("bil_h"): return "bil"
    if base.startswith("sov_h"): return "sov"
    return base


def is_equivariant(name):
    return not parse_name(name)[0].startswith("flat")


# ============================================================ training
def full_batch_step(model, X, y, opt):
    opt.zero_grad()
    N = len(y)
    for i in range(0, N, CHUNK):
        loss = F.cross_entropy(model(X[i:i + CHUNK]), y[i:i + CHUNK], reduction="sum") / N
        loss.backward()
    opt.step()


@torch.no_grad()
def eval_split(model, s):
    model.eval()
    nll = kl = 0.0
    N = len(s["y"])
    for i in range(0, N, CHUNK):
        logq = torch.log_softmax(model(s["X"][i:i + CHUNK]), dim=1)
        p = s["p"][i:i + CHUNK]
        nll += -logq.gather(1, s["y"][i:i + CHUNK, None]).sum().item()
        kl += (p * (torch.log(p + 1e-12) - logq)).sum().item()
    return nll / N, kl / N


def train_model(model, tr, va, lr, wd):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, best_ep, t0 = float("inf"), None, 0, time.time()
    ep = 0
    for ep in range(MAX_EPOCHS):
        model.train(); full_batch_step(model, tr["X"], tr["y"], opt)
        v = eval_split(model, va)[0]
        if v < best:
            best, best_ep, best_state = v, ep, copy.deepcopy(model.state_dict())
        if ep - best_ep >= PATIENCE: break
    model.load_state_dict(best_state)
    epochs = ep + 1
    return dict(epochs=epochs, best_epoch=best_ep, val_nll=best,
                budget_hit=bool(epochs == MAX_EPOCHS and best_ep > MAX_EPOCHS - PATIENCE),
                train_seconds=time.time() - t0)


def score_fn_of(model, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    @torch.no_grad()
    def run(X):
        model.eval()
        out = [model(torch.tensor(X[i:i + CHUNK], dtype=torch.float32, device=DEVICE))
               for i in range(0, len(X), CHUNK)]
        return torch.cat(out).double().cpu().numpy()
    def f(X):
        if not shuffle: return run(X)
        perm = np.argsort(rng.random(X.shape[:2]), axis=1)
        sp = run(np.take_along_axis(X, perm[:, :, None], axis=1))
        s = np.empty_like(sp); np.put_along_axis(s, perm, sp, axis=1)
        return s
    return f


# ============================================================ probe (memory-light) and Metric C
def run_probe(score_fn, d, seed=7, shifted=False, oversample=10):
    """Per probe pair we keep only d x d numbers (the pre-flight kept d^2 x d^2).
    The regression matrix has Kronecker structure, (delta delta^T) (x) (Y^T Y), so the fit is identical."""
    rng = np.random.default_rng(seed)
    K = PROBE_K_PER_D * d
    st = dict(S=np.zeros((K, d, d)), t=np.zeros((K, d)), ss=np.zeros(K),
              mg=np.zeros(K), Yb=np.zeros((K, d)), Dl=np.zeros((K, d)))
    for k in range(K):
        x_a, x_b = rng.normal(size=(2, d))
        n_gen = PROBE_N * (oversample if shifted else 1)
        X = np.concatenate([np.broadcast_to(x_a, (n_gen, 1, d)), np.broadcast_to(x_b, (n_gen, 1, d)),
                            rng.normal(size=(n_gen, SLATE - 2, d))], axis=1)
        y = X.mean(axis=1)
        if shifted:
            keep = np.argsort(np.linalg.norm(y, axis=1))[-PROBE_N:]
            X, y = X[keep], y[keep]
        sc = score_fn(X); gap = sc[:, 0] - sc[:, 1]
        y_bar, gap_bar = y.mean(axis=0), gap.mean()
        yc, g = y - y_bar, gap - gap_bar
        st["S"][k], st["t"][k], st["ss"][k] = yc.T @ yc, yc.T @ g, g @ g
        st["mg"][k], st["Yb"][k], st["Dl"][k] = gap_bar, y_bar, x_a - x_b
    return st


def fit_from_stats(st, idx):
    K, d = st["Dl"].shape
    w = np.bincount(idx, minlength=K).astype(float)
    D = st["Dl"]
    left = (w[:, None, None] * D[:, :, None] * D[:, None, :]).reshape(K, d * d)      # (k, p r)
    G = (left.T @ st["S"].reshape(K, d * d)).reshape(d, d, d, d)                       # (p r, q s)
    G = G.transpose(0, 2, 1, 3).reshape(d * d, d * d)                                  # (p q, r s)
    b = np.einsum("k,kp,kq->pq", w, D, st["t"]).reshape(d * d)
    ss = w @ st["ss"]
    vecA = np.linalg.lstsq(G, b, rcond=None)[0]
    A_hat = vecA.reshape(d, d)
    Dl = D[idx]
    c = st["mg"][idx] - np.einsum("kp,pq,kq->k", Dl, A_hat, st["Yb"][idx])
    th = np.linalg.lstsq(Dl, c, rcond=None)[0]
    fid_A = 1 - (ss - 2 * vecA @ b + vecA @ G @ vecA) / ss if ss > 1e-12 else np.nan
    res = c - Dl @ th
    fid_th = 1 - (res @ res) / (((c - c.mean()) ** 2).sum() + 1e-12)
    return A_hat, th, fid_A, fid_th


def recovery_metrics(A_hat, th_hat, A_true, th_true):
    a, b = A_hat.ravel(), A_true.ravel()
    fa, fb = np.linalg.norm(a), np.linalg.norm(b)
    out = dict(fro_hat=fa)
    if fb > 1e-12:
        out.update(cosine=a @ b / (fa * fb + 1e-12), rel_frob=np.linalg.norm(a - b) / fb, scale_ratio=fa / fb)
        zeros = A_true == 0
        if zeros.any():
            out["zero_mass"] = np.linalg.norm(A_hat[zeros]) / fb
            out["zero_ratio"] = np.abs(A_hat[zeros]).mean() / (np.abs(A_hat[~zeros]).mean() + 1e-12)
    else:
        out.update(cosine=np.nan, rel_frob=np.nan, scale_ratio=np.nan)
    out["theta_cos"] = th_hat @ th_true / (np.linalg.norm(th_hat) * np.linalg.norm(th_true) + 1e-12)
    out["theta_relerr"] = np.linalg.norm(th_hat - th_true) / np.linalg.norm(th_true)
    return out


def probe_report(score_fn, A_true, th_true, d, seed=7):
    st = run_probe(score_fn, d, seed=seed)
    K = len(st["ss"])
    A_hat, th_hat, fid_A, fid_th = fit_from_stats(st, np.arange(K))
    row = recovery_metrics(A_hat, th_hat, A_true, th_true)
    row.update(fid_A=fid_A, fid_theta=fid_th)
    brng = np.random.default_rng(seed + 999)
    boots = [recovery_metrics(*fit_from_stats(st, brng.integers(0, K, size=K))[:2], A_true, th_true)
             for _ in range(n_boot(d))]
    for m in ["fro_hat", "cosine", "theta_cos"]:
        vals = np.array([bt[m] for bt in boots], dtype=float)
        row[m + "_lo"], row[m + "_hi"] = (np.nanpercentile(vals, [2.5, 97.5]) if np.isfinite(vals).any()
                                          else (np.nan, np.nan))
    A_sh = fit_from_stats(run_probe(score_fn, d, seed=seed + 1, shifted=True), np.arange(K))[0]
    row["inv_abs"] = np.linalg.norm(A_hat - A_sh)
    row["inv_rel"] = row["inv_abs"] / (np.linalg.norm(A_hat) + 1e-12)
    return row, A_hat, th_hat


def metric_c(score_fn, A_true, A_ref, d, seed=11):
    rng = np.random.default_rng(seed)
    n = C_TRIALS
    xa, xb = rng.normal(size=(n, 1, d)), rng.normal(size=(n, 1, d))
    X1 = np.concatenate([xa, xb, rng.normal(size=(n, SLATE - 2, d))], axis=1)
    X2 = np.concatenate([xa, xb, rng.normal(size=(n, SLATE - 2, d))], axis=1)
    s1, s2 = score_fn(X1), score_fn(X2)
    d_mod = (s2[:, 0] - s2[:, 1]) - (s1[:, 0] - s1[:, 1])
    delta, dy = (xa - xb)[:, 0], X2.mean(axis=1) - X1.mean(axis=1)
    change = lambda A: np.einsum("np,pq,nq->n", delta, A, dy)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    t, t_ref = change(A_true), change(A_ref)
    out = dict(c_ratio=rms(d_mod) / (rms(t_ref) + 1e-12))
    if np.std(t) > 1e-12:
        out["c_slope"] = float(np.cov(t, d_mod)[0, 1] / np.var(t, ddof=1))
        out["c_r2"] = float(np.corrcoef(t, d_mod)[0, 1] ** 2)
    else:
        out["c_slope"] = out["c_r2"] = np.nan
    return out


def ey_ceiling(A, h):
    s = np.linalg.svd(A, compute_uv=False)
    return float(np.sqrt((s[:h] ** 2).sum() / (s ** 2).sum()))


# ============================================================ the three task types
def ladder_rows(task):
    """Known-answer checks of the probe and Metric C at a given d (no training)."""
    d, rank = task["d"], task["rank"]
    th, A = make_ground_truth(d, rank, seed=12345)
    U_, S_, Vt_ = np.linalg.svd(A)
    Wq, Wk = np.sqrt(S_)[:, None] * U_.T, np.sqrt(S_)[:, None] * Vt_
    fns = {
        "V0 context-free, nonlinear": lambda X: X @ th + np.sin(X).sum(axis=-1),
        "V1 generator's own score": lambda X: np.einsum("nd,nsd->ns", th[None, :] + X.mean(axis=1) @ A.T, X),
        "V2 hand-built attention": lambda X: X @ th + np.einsum("nid,njd->nij", X @ Wq.T, X @ Wk.T).mean(axis=2),
        "V4 saturating (not LCL)": lambda X: X @ th + 0.5 * np.tanh(2.0 * np.einsum("nd,nsd->ns", X.mean(axis=1) @ A.T, X)),
    }
    rows = []
    for name, fn in fns.items():
        r, A_hat, th_hat = probe_report(fn, A, th, d)
        r.update(metric_c(fn, A, A, d))
        r.update(rung=name, d=d, rank=rank, max_err_A=float(np.abs(A_hat - A).max()),
                 max_err_theta=float(np.abs(th_hat - th).max()))
        rows.append(r)
    return rows


def train_rows(task, hp, measure=True):
    d, rank, n = task["d"], task["rank"], task["n"]
    sparsity = task.get("sparsity", 0.0)
    seed = task["seed"]
    theta, A_ctx = make_ground_truth(d, rank, seed, sparsity=sparsity)
    A = A_ctx if task["world"] == "context" else np.zeros_like(A_ctx)
    dev = build_splits(theta, A, n, d, data_seed=seed + 1)
    rows = []
    for name in task["models"]:
        h = dict(hp.get(family(name), HP_DEFAULT))
        h.update(task.get("hp_override", {}))
        init_seed = task.get("init_seed", seed)
        torch.manual_seed(init_seed)
        model = build_model(name, d).to(DEVICE)
        row = {k: v for k, v in task.items() if k not in ("models", "hp_override")}
        row.update(model=name, family=family(name), params=n_params(model), lr=h["lr"], wd=h["wd"],
                   init_seed=init_seed)
        row.update(train_model(model, dev["train"], dev["val"], h["lr"], h["wd"]))
        if measure:
            row["nll_in"], row["kl_in"] = eval_split(model, dev["eval_in"])
            row["nll_shift"], row["kl_shift"] = eval_split(model, dev["eval_shift"])
            sf = score_fn_of(model, shuffle=not is_equivariant(name), seed=seed)
            rep, A_hat, th_hat = probe_report(sf, A, theta, d)
            row.update(rep); row.update(metric_c(sf, A, A_ctx, d))
            if True:                                # save the recovered matrices for every run (figures, identification test)
                row["A_hat"] = json.dumps(np.round(A_hat, 6).ravel().tolist())
                row["theta_hat"] = json.dumps(np.round(th_hat, 6).tolist())
            base = parse_name(name)[0]
            if base.startswith(("bil_h", "sov_h")) and task["world"] == "context":
                row["ey_ceiling"] = ey_ceiling(A, int(base[5:]))
        rows.append(row)
    return rows


def run_task(task, hp):
    if task["type"] == "ladder":
        return ladder_rows(task)
    if task["type"] == "tune":
        return train_rows(task, hp, measure=False)
    return train_rows(task, hp, measure=True)
