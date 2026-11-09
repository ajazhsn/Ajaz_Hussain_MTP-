"""
synth.py - Synthetic higher-order context-effect study (MTP, Ajaz Hussain).

Same file runs on Kaggle (import it in cells) and on the cluster (command line).
Data are regenerated deterministically from (dataset, seed), so every platform
sees identical data and no data files need to be copied around.

Datasets (first set of experiments, "pure" orders):
  D0_lcl    : LCL,  u_i = theta.x_i + lam * x_i' A ybar_S           (order 1, linear)
  D1_sim    : similarity, closeness of i to ONE other item           (order 1, nonlinear)
  D2_comp   : compromise, closeness of i to the MIDPOINT of TWO      (order 2)
  D3_centre : centre, closeness of i to the CENTRE of THREE          (order 3)
"""
import argparse, copy, itertools, json, math, os, re, time, zlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ------------------------------------------------------------------ settings
N_ITEMS, D = 10, 6            # items per slate, features per item
CTX = [0, 1]                  # the 2 features used for closeness ("price", "quality")
N_TRAIN, N_VAL, N_TEST = 285_000, 15_000, 50_000

DATASETS = {
    "D0_lcl":    dict(id=0, kind="lcl"),
    "D1_sim":    dict(id=1, kind="k", k=1, sigma=0.2, sign=-1.0),   # near-copy steals share
    "D2_comp":   dict(id=2, kind="k", k=2, sigma=0.2, sign=+1.0),   # middle option gains
    "D3_centre": dict(id=3, kind="k", k=3, sigma=0.2, sign=+1.0),
    # robustness: same effects, but "close" = inside a box window of radius r (not a Gaussian)
    "D1b_sim":    dict(id=4, kind="k", k=1, sigma=None, sign=-1.0, kernel="box"),
    "D2b_comp":   dict(id=5, kind="k", k=2, sigma=None, sign=+1.0, kernel="box"),
    "D3b_centre": dict(id=6, kind="k", k=3, sigma=None, sign=+1.0, kernel="box"),
    # depth-order study: same Gaussian closeness, to the centre of FOUR / FIVE other items
    "D4_quad":    dict(id=7, kind="k", k=4, sigma=0.2, sign=+1.0),
    "D5_quint":   dict(id=8, kind="k", k=5, sigma=0.2, sign=+1.0),
}
# Strength of the planted context effect, from Exp 0 (Cell 0.5), sigma = 0.2:
#   D0: twin gap 0.05 nats (control, effect is the slate mean)
#   D1-D3: part NOT explained by own + others' mean & sd = 0.03 nats in each
# Use the SAME values on every platform.
LAMBDA = {"D0_lcl": 1.3120, "D1_sim": 1.8310, "D2_comp": 0.5087, "D3_centre": 0.2507,
          "D1b_sim": None, "D2b_comp": None, "D3b_centre": None, "D4_quad": None, "D5_quint": None}

# Box-kernel radius and strengths come from calib_box.py (written to box_calib.json).
_BOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_calib.json")
if os.path.exists(_BOX):
    _b = json.load(open(_BOX))
    for _d in ("D1b_sim", "D2b_comp", "D3b_centre"):
        DATASETS[_d]["sigma"] = _b["radius"]; LAMBDA[_d] = _b["lambda"][_d]

# Strengths of the order-4/5 datasets come from calib_order.py (written to order_calib.json).
_ORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_calib.json")
if os.path.exists(_ORD):
    for _d, _l in json.load(open(_ORD))["lambda"].items():
        LAMBDA[_d] = _l


# ------------------------------------------------------------------ ground truth
def truth(ds, seed):
    """theta (and A for LCL) drawn per seed; ||theta|| = 1, ||A||_F = 1."""
    rng = np.random.default_rng([DATASETS[ds]["id"], seed, 1])
    theta = rng.normal(size=D); theta /= np.linalg.norm(theta)
    U, V = rng.normal(size=(D, 2)), rng.normal(size=(D, 2))
    A = U @ V.T; A /= np.linalg.norm(A)
    return theta, A


def context_raw(X, ds, A, chunk=10_000):
    """Planted context term c_i (before multiplying by lambda). X: (N, n, D)."""
    spec = DATASETS[ds]
    if spec["kind"] == "lcl":
        ybar = X.mean(1, keepdims=True)                       # mean of the whole slate
        return ((X @ A) * ybar).sum(-1)                       # x_i' A ybar
    k, s2 = spec["k"], spec["sigma"] ** 2
    combos = np.array(list(itertools.combinations(range(N_ITEMS), k)))   # (T, k)
    member = np.zeros((N_ITEMS, len(combos)), bool)
    for t, c in enumerate(combos):
        member[c, t] = True                                    # item i is inside combo t
    out = np.empty(X.shape[:2])
    for a in range(0, len(X), chunk):
        P = X[a:a + chunk][..., CTX]                           # (B, n, 2)
        cent = P[:, combos, :].mean(2)                         # (B, T, 2) midpoint / centre
        d2 = ((P[:, :, None, :] - cent[:, None, :, :]) ** 2).sum(-1)   # (B, n, T)
        K = (d2 < s2) if spec.get("kernel") == "box" else np.exp(-d2 / s2)
        K = K * (~member)[None]                                # only combos of OTHER items
        out[a:a + chunk] = K.sum(-1)
    return out


def make_data(ds, seed, lam=None, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST):
    """Returns dict with X (N,n,D) float32, y (N,), P (true probs), U (utilities), splits."""
    lam = LAMBDA[ds] if lam is None else lam
    assert lam is not None, f"LAMBDA['{ds}'] not set - run Exp 0 first"
    theta, A = truth(ds, seed)
    rng = np.random.default_rng([DATASETS[ds]["id"], seed, 2])
    N = n_train + n_val + n_test
    X = rng.normal(size=(N, N_ITEMS, D))
    c = context_raw(X, ds, A)
    sign = DATASETS[ds].get("sign", 1.0)
    u = X @ theta + sign * lam * c
    P = np.exp(u - u.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) < rng.random((N, 1))).sum(1).clip(max=N_ITEMS - 1)
    u0 = X @ theta                                             # context-free truth
    idx = {"train": slice(0, n_train), "val": slice(n_train, n_train + n_val),
           "test": slice(n_train + n_val, N)}
    return dict(X=X.astype(np.float32), y=y, P=P.astype(np.float32),
                U0=u0.astype(np.float32), c=c, idx=idx, theta=theta, A=A)


def make_data_sized(ds, seed, n_train):
    """Same val/test sets as make_data (so results are comparable across sizes), and NESTED
    training sets: n_train <= N_TRAIN uses the first n_train standard training slates;
    n_train > N_TRAIN appends extra slates drawn from a separate random stream."""
    data = make_data(ds, seed, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
    if n_train <= N_TRAIN:
        data["idx"]["train"] = slice(0, n_train); return data
    theta, A = data["theta"], data["A"]
    rng = np.random.default_rng([DATASETS[ds]["id"], seed, 5])
    E = n_train - N_TRAIN
    Xe = rng.normal(size=(E, N_ITEMS, D)); ce = context_raw(Xe, ds, A)
    ue = Xe @ theta + DATASETS[ds].get("sign", 1.0) * LAMBDA[ds] * ce
    Pe = np.exp(ue - ue.max(1, keepdims=True)); Pe /= Pe.sum(1, keepdims=True)
    ye = (Pe.cumsum(1) < rng.random((E, 1))).sum(1).clip(max=N_ITEMS - 1)
    t, rest = slice(0, N_TRAIN), slice(N_TRAIN, None)
    cat = lambda a, b: np.concatenate([a[t], b, a[rest]])
    out = dict(X=cat(data["X"], Xe.astype(np.float32)), y=cat(data["y"], ye), P=cat(data["P"], Pe.astype(np.float32)),
               U0=cat(data["U0"], (Xe @ theta).astype(np.float32)), c=cat(data["c"], ce), theta=theta, A=A)
    out["idx"] = {"train": slice(0, n_train), "val": slice(n_train, n_train + N_VAL),
                  "test": slice(n_train + N_VAL, n_train + N_VAL + N_TEST)}
    return out

def oracle_metrics(data):
    """Best possible numbers (true model) and the context-free truth, on test."""
    t = data["idx"]["test"]; P, y = data["P"][t].astype(np.float64), data["y"][t]
    logP = np.log(P + 1e-30)
    U0 = data["U0"][t].astype(np.float64)
    logP0 = U0 - U0.max(1, keepdims=True); logP0 -= np.log(np.exp(logP0).sum(1, keepdims=True))
    return dict(test_nll=float(-logP[np.arange(len(y)), y].mean()),
                test_ce=float(-(P * logP).sum(1).mean()),            # = entropy
                plain_test_ce=float(-(P * logP0).sum(1).mean()))      # theta only, no context


# ------------------------------------------------------------------ Exp 0: calibration
def calibrate(ds, lams, seeds=(0, 1, 2), n_slates=20_000, target=0.05):
    """No training. For each lambda, estimate how much the planted context is worth
    to a no-context model ("twin gap") and how much of it simple slate stats
    (own + others' mean & sd) can already explain ("low-order share")."""
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR
    rows = []
    for seed in seeds:
        theta, A = truth(ds, seed)
        rng = np.random.default_rng([99, DATASETS[ds]["id"], seed])
        X = rng.normal(size=(n_slates, N_ITEMS, D)); c = context_raw(X, ds, A)
        cc = c - c.mean(1, keepdims=True)                       # only within-slate differences matter
        S = X.sum(1, keepdims=True); S2 = (X ** 2).sum(1, keepdims=True)
        m = (S - X) / (N_ITEMS - 1); sd = np.sqrt(np.clip((S2 - X ** 2) / (N_ITEMS - 1) - m ** 2, 0, None))
        f_own = X.reshape(-1, D); f_low = np.concatenate([X, m, sd], -1).reshape(-1, 3 * D)
        h = len(f_own) // 2                                     # fit on half, predict on other half
        pred = {}
        for name, f in [("own", f_own), ("low", f_low)]:
            g = GBR(max_iter=300, learning_rate=0.1).fit(f[:h], cc.reshape(-1)[:h])
            pred[name] = g.predict(f[h:]).reshape(-1, N_ITEMS)
        Xh, cch = X.reshape(-1, D)[h:].reshape(-1, N_ITEMS, D), cc.reshape(-1)[h:].reshape(-1, N_ITEMS)
        r2 = {k: 1 - ((cch - v) ** 2).mean() / (cch ** 2).mean() for k, v in pred.items()}
        base = Xh @ theta; sign = DATASETS[ds].get("sign", 1.0)

        def ce(u_model, u_true):
            lp = u_model - u_model.max(1, keepdims=True); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
            pt = np.exp(u_true - u_true.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)
            return -(pt * lp).sum(1).mean()
        for lam in lams:
            ut = base + sign * lam * cch
            o = ce(ut, ut)
            rows.append(dict(seed=seed, lam=lam, r2_own=r2["own"], r2_low=r2["low"],
                             gap_twin=ce(base + sign * lam * pred["own"], ut) - o,
                             gap_low=ce(base + sign * lam * pred["low"], ut) - o,
                             gap_plain=ce(base, ut) - o))
    return rows


# ------------------------------------------------------------------ models (input X: (B, n, D) -> logits (B, n))
def mlp(din, w=64, out=1):
    return nn.Sequential(nn.Linear(din, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, out))

class PerItem(nn.Module):                       # no context at all
    def __init__(self, w=64): super().__init__(); self.f = mlp(D, w)
    def forward(self, X): return self.f(X).squeeze(-1)

def others_stats(X):
    n = X.shape[1]; S = X.sum(1, keepdims=True); S2 = (X ** 2).sum(1, keepdims=True)
    m = (S - X) / (n - 1); var = ((S2 - X ** 2) / (n - 1) - m ** 2).clamp_min(0)
    return m, (var + 1e-6).sqrt()

class PoolMean(nn.Module):                      # ladder rung 1: own + mean of others
    def __init__(self, w=64): super().__init__(); self.f = mlp(2 * D, w)
    def forward(self, X): m, _ = others_stats(X); return self.f(torch.cat([X, m], -1)).squeeze(-1)

class PoolMeanSD(nn.Module):                    # ladder rung 2: + spread of others
    def __init__(self, w=64): super().__init__(); self.f = mlp(3 * D, w)
    def forward(self, X): m, s = others_stats(X); return self.f(torch.cat([X, m, s], -1)).squeeze(-1)

class SelfAttn(nn.Module):                      # thesis model (post-LN encoder, as in notebook)
    def __init__(self, width=32, heads=2, ffn=64, layers=1, self_only=False):
        super().__init__(); self.self_only = self_only
        self.embed = nn.Linear(D, width); self.ln0 = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            attn=nn.MultiheadAttention(width, heads, batch_first=True), ln1=nn.LayerNorm(width),
            ffn=nn.Sequential(nn.Linear(width, ffn), nn.ReLU(), nn.Linear(ffn, width)), ln2=nn.LayerNorm(width)))
            for _ in range(layers)])
        self.out = nn.Linear(width, 1)
    def forward(self, X):
        h = self.ln0(self.embed(X)); n = X.shape[1]
        mask = ~torch.eye(n, dtype=torch.bool, device=X.device) if self.self_only else None
        for b in self.blocks:
            h = b["ln1"](h + b["attn"](h, h, h, attn_mask=mask, need_weights=False)[0])
            h = b["ln2"](h + b["ffn"](h))
        return self.out(h).squeeze(-1)

class AttnOnly(nn.Module):
    """Depth-order study. Per-item MLP embedding, then L attention layers with residual
    connections ONLY (no feed-forward, no LayerNorm between layers), then a linear readout.
    kind='soft': softmax attention.  kind='lin': linear attention, a_ij = q_i.k_j / (sqrt(dh) n),
    so the utility is a polynomial in the other items' embeddings whose interaction order is at
    most 2^L - 1 (the per-item MLP cannot raise the order because it acts before any mixing)."""
    def __init__(self, width=32, heads=2, layers=1, kind="soft"):
        super().__init__(); self.kind, self.H = kind, heads
        self.embed = nn.Sequential(nn.Linear(D, width), nn.ReLU(), nn.Linear(width, width))
        self.qkv = nn.ModuleList([nn.Linear(width, 3 * width) for _ in range(layers)])
        self.o = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        if kind == "lin":                                   # start small: products of layers can blow up
            for o in self.o: nn.init.normal_(o.weight, std=0.1 / math.sqrt(width)); nn.init.zeros_(o.bias)
        self.out = nn.Linear(width, 1)
    def forward(self, X):
        h = self.embed(X); B, n, d = h.shape; H = self.H; dh = d // H
        for qkv, o in zip(self.qkv, self.o):
            q, k, v = (t.view(B, n, H, dh).transpose(1, 2) for t in qkv(h).split(d, -1))
            s = q @ k.transpose(-1, -2) / math.sqrt(dh)
            w = torch.softmax(s, -1) if self.kind == "soft" else s / n
            h = h + o((w @ v).transpose(1, 2).reshape(B, n, d))
        return self.out(h).squeeze(-1)

# --- DeepHalo: copied unchanged from the authors' DeepHalo-Code/Model/FeatureBased.py
def make_valid_mask(n, lengths, device, mode='without'):
    row = torch.arange(n, device=device).unsqueeze(0)
    if mode == 'with':
        return (row < lengths.unsqueeze(1)) | (row == n - 1)
    else:
        return row < lengths.unsqueeze(1)

def masked_softmax(scores, lengths):
    n = scores.size(1)
    valid = make_valid_mask(n, lengths, scores.device)
    scores = scores.masked_fill(~valid, float('-inf'))
    return F.log_softmax(scores, dim=-1)

class NonlinearTransformation(nn.Module):
    def __init__(self, H, embed=128, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(embed, embed * H)
        self.fc2 = nn.Linear(embed, embed)
        self.H = H
        self.embed = embed
        self.dropout = nn.Dropout(dropout)
        self.enc_norm = nn.LayerNorm(embed)
    def forward(self, X):
        B, n, _ = X.shape
        X = self.fc1(X).view(B, n, self.H, self.embed)
        X = nn.ReLU()(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.enc_norm(X)
        return X

class DeepHalo(nn.Module):
    def __init__(self, n, input_dim, H, L, embed=128, dropout=0):
        super().__init__()
        self.basic_encoder = nn.Sequential(
            nn.Linear(input_dim, embed), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed, embed), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed, embed))
        self.enc_norm = nn.LayerNorm(embed)
        self.aggregate_linear = nn.ModuleList([nn.Linear(embed, H) for _ in range(L)])
        self.nonlinear = nn.ModuleList([NonlinearTransformation(H, embed) for _ in range(L)])
        self.H = H
        self.embed = embed
        self.final_linear = nn.Linear(embed, 1)
        self.qualinear1 = nn.Linear(embed, embed)
        self.qualinear2 = nn.Linear(embed, embed)
    def forward(self, X, lengths):
        B, n, _ = X.shape
        Z = self.enc_norm(self.basic_encoder(X))
        X = Z.clone()
        for fc, nt in zip(self.aggregate_linear, self.nonlinear):
            Z_bar = (fc(Z).sum(1) / lengths.unsqueeze(1)).unsqueeze(-1).unsqueeze(1)
            phi = nt(X)
            valid = make_valid_mask(n, lengths, X.device)
            phi = phi * valid.unsqueeze(-1).unsqueeze(-1)
            Z = (phi * Z_bar).sum(2) / self.H + Z
        logits = self.final_linear(Z).squeeze(-1)
        probs = masked_softmax(logits, lengths)
        return probs, logits
# --- end of authors' code

class DH(nn.Module):                            # thin wrapper: full slates, return logits
    def __init__(self, L=4, embed=32, H=8):     # paper's LPMC config: embed 32, H 8, L 4
        super().__init__(); self.net = DeepHalo(N_ITEMS, D, H, L, embed)
    def forward(self, X):
        lengths = torch.full((X.shape[0],), X.shape[1], device=X.device, dtype=torch.float32)
        return self.net(X, lengths)[1]

class DHQ(DH):
    """DeepHalo with the paper's quadratic ("Hadamard product") aggregation, eq. (6):
       Z_bar = mean_k W sigma(z_k), sigma(z) = qualinear1(z) * qualinear2(z) (element-wise).
    The released code defines qualinear1/2 but never uses them; this wires them in.
    Everything else (encoder, phi, residual, final layer) is the authors' code."""
    uses_q = True
    def forward(self, X):
        net = self.net; n = X.shape[1]
        Z = net.enc_norm(net.basic_encoder(X)); X0 = Z.clone()
        for fc, nt in zip(net.aggregate_linear, net.nonlinear):
            Zq = net.qualinear1(Z) * net.qualinear2(Z)                       # quadratic sigma
            Z_bar = fc(Zq).mean(1).unsqueeze(-1).unsqueeze(1)                 # (B, 1, H, 1)
            Z = (nt(X0) * Z_bar).sum(2) / net.H + Z
        return net.final_linear(Z).squeeze(-1)

MODELS = {
    "mlp":         lambda: PerItem(),
    "sa1_self":    lambda: SelfAttn(layers=1, self_only=True),   # blindfolded twin of sa1
    "pool_mean":   lambda: PoolMean(),
    "pool_meansd": lambda: PoolMeanSD(),
    "sa1":         lambda: SelfAttn(layers=1),
    "sa2":         lambda: SelfAttn(layers=2),
    "dh1":         lambda: DH(L=1),
    "dh2":         lambda: DH(L=2),
    "dh3":         lambda: DH(L=3),
    "dh4":         lambda: DH(L=4),                               # DeepHalo "pure": paper config
    # --- size controls (appended at the end so the seeds of the models above do not change)
    "sa1_wide":    lambda: SelfAttn(width=46, ffn=92, layers=1),  # 1 layer at sa2's size (~17.9k)
    "sa2_small":   lambda: SelfAttn(width=22, ffn=44, layers=2),  # 2 layers at sa1's size (~8.4k)
    "dh_s":        lambda: DH(L=4, embed=16, H=6),                # DeepHalo at sa1's size (~8.9k)
    "dh_big":      lambda: DH(L=4, embed=64, H=8),                # DeepHalo 4x wider (~161k)
    # --- DeepHalo, paper's quadratic (Hadamard) variant
    "dhq2":        lambda: DHQ(L=2),                              # order <= 2
    "dhq4":        lambda: DHQ(L=4),                              # paper config (embed 32, H 8, L 4)
    "dhq_s":       lambda: DHQ(L=4, embed=16, H=6),               # at sa1's size (~9.4k)
}

def get_model(name):
    """Named models from MODELS, plus tuning-grid names:
       sa_L{layers}_w{width}_h{heads}   (ffn = 2*width)
       dhq_L{layers}_e{embed}_H{H}      (paper's quadratic DeepHalo)
       aosoft_L{L}_w{w}_h{h} / aolin_L{L}_w{w}_h{h}   (attention-only, softmax / linear)"""
    if name in MODELS: return MODELS[name]()
    m = re.fullmatch(r"sa_L(\d+)_w(\d+)_h(\d+)", name)
    if m:
        L, w, h = map(int, m.groups()); return SelfAttn(width=w, heads=h, ffn=2 * w, layers=L)
    m = re.fullmatch(r"dhq_L(\d+)_e(\d+)_H(\d+)", name)
    if m:
        L, e, H = map(int, m.groups()); return DHQ(L=L, embed=e, H=H)
    m = re.fullmatch(r"ao(soft|lin)_L(\d+)_w(\d+)_h(\d+)", name)     # attention-only (depth-order study)
    if m:
        kind = m.group(1); L, w, h = map(int, m.groups()[1:]); return AttnOnly(width=w, heads=h, layers=L, kind=kind)
    raise KeyError(name)

def model_index(name):          # seed offset per model; unchanged for the original models
    return list(MODELS).index(name) if name in MODELS else zlib.crc32(name.encode()) % 997 + 100

def n_params(m, used_only=True):
    skip = ("qualinear",) if used_only and not getattr(m, "uses_q", False) else ()
    return sum(p.numel() for n, p in m.named_parameters() if not any(s in n for s in skip))


# ------------------------------------------------------------------ training / evaluation
def to_tensors(data, dev):
    T = {}
    for s, ix in data["idx"].items():
        T[s] = (torch.tensor(data["X"][ix], device=dev), torch.tensor(data["y"][ix], device=dev),
                torch.tensor(data["P"][ix], device=dev))
    return T

@torch.no_grad()
def evaluate(model, T, split, bs=8192):
    X, y, P = T[split]; model.eval(); nll = ce = acc = 0.0
    for i in range(0, len(y), bs):
        lp = F.log_softmax(model(X[i:i + bs]), -1)
        nll += -lp.gather(1, y[i:i + bs, None]).sum().item()
        ce += -(P[i:i + bs] * lp).sum().item()
        acc += (lp.argmax(1) == y[i:i + bs]).sum().item()
    return dict(nll=nll / len(y), ce=ce / len(y), acc=acc / len(y))

def train(model, T, seed, bs=512, phases=((1e-3, 6), (1e-4, 4)), max_epochs=80, log=None):
    """Adam, early stopping on validation NLL; second phase at lower lr from the best state."""
    g = torch.Generator().manual_seed(seed); X, y, _ = T["train"]; n = len(y)
    best_v, best = evaluate(model, T, "val")["nll"], copy.deepcopy(model.state_dict()); eps = []
    for lr, patience in phases:
        model.load_state_dict(best); opt = torch.optim.Adam(model.parameters(), lr=lr); wait = 0
        for ep in range(max_epochs):
            model.train(); perm = torch.randperm(n, generator=g).to(X.device)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                loss = F.cross_entropy(model(X[b]), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
            v = evaluate(model, T, "val")["nll"]
            if log: log(f"    lr {lr:g} ep {ep + 1:3d} val {v:.5f}")
            if v < best_v - 1e-5: best_v, best, wait = v, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience: break
        eps.append(ep + 1)
    model.load_state_dict(best)
    return model, eps


# ------------------------------------------------------------------ sweep runner (resumable)
def run_jobs(datasets, models, seeds, out, dev, shard=(0, 1), verbose=False, save_weights=False, hours=None,
             phases=((1e-3, 6), (1e-4, 4)), n_train=None, max_epochs=80):
    """Every (dataset, seed) pair in this shard: generate data once, write the oracle
    file, then train each model. Finished runs (json exists) are skipped, so the
    same command can be re-launched after a crash/timeout on any platform.
    hours: stop starting new runs after this many hours (so Kaggle/SLURM end cleanly)."""
    deadline = time.time() + 3600 * hours if hours else None
    os.makedirs(f"{out}/results", exist_ok=True)
    if save_weights: os.makedirs(f"{out}/weights", exist_ok=True)
    pairs = [(d, s) for d in datasets for s in seeds][shard[0]::shard[1]]
    for ds, seed in pairs:
        todo = [m for m in models if not os.path.exists(f"{out}/results/{ds}__{m}__s{seed}.json")]
        ofile = f"{out}/results/{ds}__oracle__s{seed}.json"
        if not todo and os.path.exists(ofile): continue
        t0 = time.time(); data = make_data(ds, seed) if n_train is None else make_data_sized(ds, seed, n_train)
        if not os.path.exists(ofile):
            json.dump(dict(dataset=ds, model="oracle", seed=seed, lam=LAMBDA[ds], **oracle_metrics(data)),
                      open(ofile, "w"))
        T = to_tensors(data, dev); gen_s = time.time() - t0; del data
        for m in todo:
            if deadline and time.time() > deadline:
                print("time budget used up - stopping cleanly; rerun the same command to continue", flush=True)
                return
            torch.manual_seed(1000 * seed + model_index(m))
            model = get_model(m).to(dev); t1 = time.time()
            model, eps = train(model, T, seed, phases=phases, max_epochs=max_epochs, log=print if verbose else None)
            r = dict(dataset=ds, model=m, seed=seed, lam=LAMBDA[ds], params=n_params(model), phases=phases, n_train=n_train or N_TRAIN,
                     epochs=eps, train_s=round(time.time() - t1, 1), gen_s=round(gen_s, 1),
                     **{f"val_{k}": v for k, v in evaluate(model, T, "val").items()},
                     **{f"test_{k}": v for k, v in evaluate(model, T, "test").items()})
            json.dump(r, open(f"{out}/results/{ds}__{m}__s{seed}.json", "w"))
            if save_weights: torch.save(model.state_dict(), f"{out}/weights/{ds}__{m}__s{seed}.pt")
            print(f"{ds} s{seed} {m:12s} test_ce {r['test_ce']:.5f} nll {r['test_nll']:.5f} "
                  f"params {r['params']} epochs {eps} {r['train_s']}s", flush=True)

def time_models(ds="D3_centre", models=None, dev="cpu", seed=0):
    """Exp 1: data-generation time and seconds per training epoch for each model."""
    t0 = time.time(); data = make_data(ds, seed); gen = time.time() - t0
    T = to_tensors(data, dev); del data
    print(f"data generation ({ds}): {gen:.1f}s   device {dev}   threads {torch.get_num_threads()}", flush=True)
    X, y, _ = T["train"]; n = len(y); rows = []
    for m in (models or list(MODELS)):
        model = get_model(m).to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3)
        model.train(); perm = torch.randperm(n).to(dev); t1 = time.time()
        for i in range(0, n, 512):
            b = perm[i:i + 512]; loss = F.cross_entropy(model(X[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        evaluate(model, T, "val")
        if dev == "cuda": torch.cuda.synchronize()
        rows.append((m, n_params(model), time.time() - t1))
        print(f"  {m:12s} params {rows[-1][1]:6d}   {rows[-1][2]:6.1f} s/epoch", flush=True)
    return rows

def collect(out):
    import pandas as pd
    rows = [json.load(open(f"{out}/results/{f}")) for f in os.listdir(f"{out}/results") if f.endswith(".json")]
    return pd.DataFrame(rows)

def parse_seeds(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--models", default="")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--out", default="synth_out")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--patience", default="6,4", help="early-stopping patience for lr 1e-3 and 1e-4")
    ap.add_argument("--n-train", type=int, default=None, help="training slates (nested; val/test unchanged)")
    ap.add_argument("--max-epochs", type=int, default=80)
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    i, k = map(int, a.shard.split("/"))
    print(f"device {dev} threads {torch.get_num_threads()} shard {i}/{k}", flush=True)
    if a.timing:
        time_models(models=a.models.split(",") if a.models else None, dev=dev)
    else:
        run_jobs(a.datasets.split(","), a.models.split(","), parse_seeds(a.seeds), a.out, dev, (i, k), a.verbose, hours=a.hours,
                 phases=tuple(zip((1e-3, 1e-4), map(int, a.patience.split(",")))),
                 n_train=a.n_train, max_epochs=a.max_epochs)
