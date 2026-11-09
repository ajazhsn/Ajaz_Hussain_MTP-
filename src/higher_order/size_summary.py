"""
interpret.py - a first look inside the trained models (MTP, Ajaz Hussain).

Reads the weights saved by interp_train.py (synth_interp/weights) and runs four analyses
on the test slates. Works on whatever weights exist, so it can be run before all
training has finished.

  0. Check       : the retrained models reproduce the final-run test loss (synth_final/).
  1. Recover A   : on LCL data (D0), fit the LCL form  u_i = theta.x_i + x_i' A xbar_S  to the
                   MODEL's utilities and compare the fitted theta, A with the true ones.
  2. Where       : (self-attention only) switch each layer's attention to 'uniform'
                   (plain average of all items) or 'self' (each item sees only itself)
                   and measure how much test loss rises; also the shape of the attention.
  3. IIA probe   : keep items 0 and 1 of a slate, replace the other 8, and compare the
                   change in u_0 - u_1 under the truth and under the model.

Usage:  python interpret.py            (writes interp_out/interp_results.json and prints tables)
"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
import synth

D, N = synth.D, synth.N_ITEMS


# ------------------------------------------------------------------ helpers
def load_model(ds, m, seed, wdir, dev):
    f = f"{wdir}/{ds}__{m}__s{seed}.pt"
    if not os.path.exists(f): return None
    model = synth.get_model(m).to(dev)
    model.load_state_dict(torch.load(f, map_location=dev)); model.eval()
    return model


@torch.no_grad()
def model_util(model, X, dev, bs=8192):
    return np.concatenate([model(torch.tensor(X[i:i + bs], device=dev)).float().cpu().numpy()
                           for i in range(0, len(X), bs)])


def true_util(ds, X, theta, A):
    c = synth.context_raw(X.astype(np.float64), ds, A)
    return X.astype(np.float64) @ theta + synth.DATASETS[ds].get("sign", 1.0) * synth.LAMBDA[ds] * c


def ce(u, P):
    """Cross-entropy of model utilities u against the true choice probabilities P (lower is better)."""
    u = u.astype(np.float64); lp = u - u.max(1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    return float(-(P.astype(np.float64) * lp).sum(1).mean())


def final_ce(final_dir, ds, m, seed):
    f = f"{final_dir}/results/{ds}__{m}__s{seed}.json"
    return json.load(open(f))["test_ce"] if os.path.exists(f) else None


# ------------------------------------------------------------------ 1. recover A (LCL)
def lcl_fit(u, X):
    """Least-squares fit of u_i = theta.x_i + x_i' A xbar_S (+ a free constant per slate,
    because utilities are only defined up to a shift within each slate)."""
    xbar = X.mean(1, keepdims=True)                                       # mean of the whole slate, as in the truth
    F2 = (X[..., :, None] * xbar[..., None, :]).reshape(*X.shape[:2], D * D)   # x_ip * xbar_q  ->  A[p, q]
    Fm = np.concatenate([X, F2], -1).astype(np.float64)
    Fm = Fm - Fm.mean(1, keepdims=True); uc = u - u.mean(1, keepdims=True)  # remove the per-slate constant
    Fm, uc = Fm.reshape(-1, Fm.shape[-1]), uc.reshape(-1)
    coef, *_ = np.linalg.lstsq(Fm, uc, rcond=None)
    r2 = 1 - ((uc - Fm @ coef) ** 2).sum() / (uc ** 2).sum()
    return coef[:D], coef[D:].reshape(D, D), float(r2)


def compare_lcl(theta_hat, A_hat, theta, A_true):
    a, b = A_hat.ravel(), A_true.ravel()
    return dict(A_corr=float(np.corrcoef(a, b)[0, 1]),
                A_slope=float(a @ b / (b @ b)),                               # 1 = right size
                A_relerr=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                theta_cos=float(theta_hat @ theta / np.linalg.norm(theta_hat) / np.linalg.norm(theta)),
                theta_slope=float(theta_hat @ theta / (theta @ theta)))


# ------------------------------------------------------------------ 2. inside self-attention
@torch.no_grad()
def sa_forward(model, X, modes):
    """Re-implements SelfAttn.forward with a choice per layer:
       'soft'    = attention as trained,
       'uniform' = every item attends equally to all items (plain average),
       'self'    = every item attends only to itself (no information from other items)."""
    h = model.ln0(model.embed(X)); B, n, d = h.shape; weights = []
    for blk, mode in zip(model.blocks, modes):
        at = blk["attn"]; H = at.num_heads; dh = d // H
        q, k, v = F.linear(h, at.in_proj_weight, at.in_proj_bias).split(d, -1)
        q, k, v = (t.view(B, n, H, dh).transpose(1, 2) for t in (q, k, v))     # (B, H, n, dh)
        if mode == "soft":
            w = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5, -1)
        elif mode == "uniform":
            w = torch.full((B, H, n, n), 1.0 / n, device=X.device)
        elif mode == "self":
            w = torch.eye(n, device=X.device).expand(B, H, n, n)
        else:
            raise ValueError(mode)
        o = at.out_proj((w @ v).transpose(1, 2).reshape(B, n, d))
        h = blk["ln1"](h + o); h = blk["ln2"](h + blk["ffn"](h))
        weights.append(w)
    return model.out(h).squeeze(-1), weights


def sa_util(model, X, dev, modes, bs=8192):
    out = []
    for i in range(0, len(X), bs):
        u, _ = sa_forward(model, torch.tensor(X[i:i + bs], device=dev), modes)
        out.append(u.float().cpu().numpy())
    return np.concatenate(out)


def attention_shape(model, X, dev, n_slates=2000):
    """Per layer and head: how spread out the attention is (1 = perfectly uniform, 0 = one item),
    the weight an item puts on itself, and whether attention follows closeness in the two
    context features (CTX) versus closeness in the other features (a control)."""
    Xt = torch.tensor(X[:n_slates], device=dev)
    _, ws = sa_forward(model, Xt, ["soft"] * len(model.blocks))
    Xc, Xo = X[:n_slates][..., synth.CTX], X[:n_slates][..., [j for j in range(D) if j not in synth.CTX]]
    near_c = -np.sqrt(((Xc[:, :, None] - Xc[:, None]) ** 2).sum(-1))           # (S, n, n): higher = closer
    near_o = -np.sqrt(((Xo[:, :, None] - Xo[:, None]) ** 2).sum(-1))
    off = ~np.eye(N, dtype=bool)
    rows = []
    for li, w in enumerate(ws):
        w = w.cpu().numpy()                                                     # (S, H, n, n)
        for hh in range(w.shape[1]):
            wh = w[:, hh]
            ent = -(wh * np.log(wh + 1e-12)).sum(-1).mean() / np.log(N)
            wo = wh[:, off]                                                     # weights on OTHER items
            rows.append(dict(layer=li + 1, head=hh + 1, spread=float(ent),
                             self_weight=float(np.diagonal(wh, axis1=1, axis2=2).mean()),
                             corr_ctx=float(np.corrcoef(wo.ravel(), near_c[:, off].ravel())[0, 1]),
                             corr_other=float(np.corrcoef(wo.ravel(), near_o[:, off].ravel())[0, 1])))
    return rows


# ------------------------------------------------------------------ 3. IIA probe
def iia_probe(u_fn, ds, X, theta, A, seed, n=20000):
    """Fix items 0 and 1, draw 8 new other items. Under IIA, u_0 - u_1 cannot change.
    Compare the change under the truth (dt) with the change under the model (dm)."""
    rng = np.random.default_rng([4242, synth.DATASETS[ds]["id"], seed])
    X1 = X[:n].copy(); X2 = X1.copy()
    X2[:, 2:, :] = rng.normal(size=(n, N - 2, D)).astype(np.float32)
    ut1, ut2 = true_util(ds, X1, theta, A), true_util(ds, X2, theta, A)
    um1, um2 = u_fn(X1), u_fn(X2)
    dt = (ut2[:, 0] - ut2[:, 1]) - (ut1[:, 0] - ut1[:, 1])
    dm = (um2[:, 0] - um2[:, 1]) - (um1[:, 0] - um1[:, 1])
    if dm.std() < 1e-9:
        return dict(iia_corr=float("nan"), iia_slope=0.0, iia_size_ratio=0.0)
    return dict(iia_corr=float(np.corrcoef(dt, dm)[0, 1]),
                iia_slope=float(np.cov(dt, dm)[0, 1] / dt.var(ddof=1)),        # 1 = right size and direction
                iia_size_ratio=float(dm.std() / dt.std()))


# ------------------------------------------------------------------ main
def summarise(rows, keys, by):
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty: return "  (no rows)"
    g = df.groupby(by)[keys]
    m, s, c = g.mean(), g.std(), g.count()
    out = m.copy().astype(object)
    for k in keys:
        out[k] = [f"{a:7.3f} ± {b / np.sqrt(n):.3f}" if n > 1 else f"{a:7.3f}"
                  for a, b, n in zip(m[k], s[k].fillna(0), c[k])]
    out["seeds"] = c[keys[0]]
    return out.to_string()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wdir", default="synth_interp/weights")
    ap.add_argument("--final", default="synth_final")
    ap.add_argument("--out", default="interp_out")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--datasets", default=None, help="comma list; default = the datasets in interp_train.py")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from interp_train import DATA, MODELS
    if a.datasets: DATA = a.datasets.split(",")
    seeds = synth.parse_seeds(a.seeds)
    R = dict(check=[], lcl=[], ablate=[], attn=[], iia=[])

    for ds in DATA:
        for seed in seeds:
            models = {m: load_model(ds, m, seed, a.wdir, dev) for m in MODELS}
            models = {m: v for m, v in models.items() if v is not None}
            if not models: continue
            data = synth.make_data(ds, seed); t = data["idx"]["test"]
            X, P, theta, A = data["X"][t], data["P"][t], data["theta"], data["A"]
            del data
            twin = final_ce(a.final, ds, "sa1_self", seed)
            print(f"{ds} seed {seed}: {sorted(models)}", flush=True)

            # truth fitted the same way (sanity check of the LCL fit itself)
            if ds == "D0_lcl" and seed == seeds[0]:
                th0, A0, r20 = lcl_fit(true_util(ds, X[:20000], theta, A), X[:20000])
                print("  LCL fit on the TRUE utilities (should be exact):",
                      compare_lcl(th0, A0, theta, synth.LAMBDA[ds] * A), f"R2 {r20:.4f}", flush=True)

            for m, model in models.items():
                u = model_util(model, X, dev); c_int = ce(u, P); c_fin = final_ce(a.final, ds, m, seed)
                R["check"].append(dict(dataset=ds, model=m, seed=seed, ce_here=c_int,
                                       ce_final=np.nan if c_fin is None else c_fin,
                                       diff=np.nan if c_fin is None else c_int - c_fin))

                if ds == "D0_lcl":                                              # 1. recover A
                    th, Ah, r2 = lcl_fit(u[:20000], X[:20000])
                    R["lcl"].append(dict(dataset=ds, model=m, seed=seed, lcl_R2=r2,
                                         **compare_lcl(th, Ah, theta, synth.LAMBDA[ds] * A)))

                fn = lambda Z, model=model: model_util(model, Z, dev)           # 3. IIA probe
                R["iia"].append(dict(dataset=ds, model=m, seed=seed, **iia_probe(fn, ds, X, theta, A, seed)))

                if m.startswith("sa"):                                          # 2. where is context computed
                    L = len(model.blocks)
                    err = np.abs(sa_util(model, X[:2000], dev, ["soft"] * L) - u[:2000]).max()
                    assert err < 1e-3, f"re-implemented forward differs from the model by {err}"
                    gain = None if twin is None else twin - c_int                 # context gain over the twin
                    for li in range(L):
                        for mode in ("uniform", "self"):
                            modes = ["soft"] * L; modes[li] = mode
                            d = ce(sa_util(model, X, dev, modes), P) - c_int
                            R["ablate"].append(dict(dataset=ds, model=m, seed=seed, layer=str(li + 1), mode=mode,
                                                    loss_rise=d, pct_of_context_gain=np.nan if not gain else 100 * d / gain))
                    for mode in ("uniform", "self"):                            # every layer at once
                        d = ce(sa_util(model, X, dev, [mode] * L), P) - c_int
                        R["ablate"].append(dict(dataset=ds, model=m, seed=seed, layer="all", mode=mode,
                                                loss_rise=d, pct_of_context_gain=np.nan if not gain else 100 * d / gain))
                    for r in attention_shape(model, X, dev):
                        R["attn"].append(dict(dataset=ds, model=m, seed=seed, **r))
            json.dump(R, open(f"{a.out}/interp_results.json", "w"), indent=1)

    print("\n=== 0. CHECK: retrained test cross-entropy minus the final run (should be ~0)")
    print(summarise(R["check"], ["ce_here", "diff"], ["dataset", "model"]))
    print("\n=== 1. RECOVER A on LCL data (D0). A_corr / theta_cos: 1 = right pattern; slopes: 1 = right size;"
          " lcl_R2: share of the model's utility explained by the LCL form")
    print(summarise(R["lcl"], ["lcl_R2", "A_corr", "A_slope", "A_relerr", "theta_cos", "theta_slope"], ["model"]))
    print("\n=== 2a. WHERE: test-loss rise when one layer's attention is replaced (pct = % of the model's context gain over its twin)")
    print(summarise([r for r in R["ablate"]], ["loss_rise", "pct_of_context_gain"], ["dataset", "model", "layer", "mode"]))
    print("\n=== 2b. ATTENTION SHAPE (spread 1 = uniform; corr_ctx > corr_other = attends to items close in the context features)")
    print(summarise(R["attn"], ["spread", "self_weight", "corr_ctx", "corr_other"], ["dataset", "model", "layer", "head"]))
    print("\n=== 3. IIA PROBE: change of u_0 - u_1 when the other 8 items are replaced (corr/slope 1 = matches the truth)")
    print(summarise(R["iia"], ["iia_corr", "iia_slope", "iia_size_ratio"], ["dataset", "model"]))
    print(f"\nall rows saved to {a.out}/interp_results.json")
