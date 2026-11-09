"""
depth_summary.py - tables for the depth-order study (results in synth_depth/).

For every dataset (order 2, 3, 4, 5) and every family (full self-attention, attention-only
softmax, attention-only linear) it prints, per depth L = 1..4, the % of the hard part captured
(beyond the mean+sd pool; 0 = pool, 100 = oracle) as mean ± s.e. over seeds, with parameters.

Usage:  python depth_summary.py [synth_depth]
"""
import sys, re
import numpy as np, pandas as pd
import synth

out = sys.argv[1] if len(sys.argv) > 1 else "synth_depth"
df = synth.collect(out)
ora = df[df.model == "oracle"].set_index(["dataset", "seed"]).test_ce
pool = df[df.model == "pool_meansd"].set_index(["dataset", "seed"]).test_ce
twin = df[df.model == "sa1_self"].set_index(["dataset", "seed"]).test_ce

def family(m):
    if m.startswith("sa_L"): return "full self-attention"
    if m.startswith("aosoft"): return "attention-only, softmax"
    if m.startswith("aolin"): return "attention-only, linear"
    return None

rows = []
for r in df.itertuples():
    f = family(r.model)
    if f is None: continue
    k = (r.dataset, r.seed)
    if k not in ora.index or k not in pool.index: continue
    hard = 100 * (pool[k] - r.test_ce) / (pool[k] - ora[k])
    gap = 100 * (twin[k] - r.test_ce) / (twin[k] - ora[k]) if k in twin.index else np.nan
    L = int(re.search(r"_L(\d+)_", r.model).group(1))
    rows.append(dict(dataset=r.dataset, family=f, L=L, model=r.model, params=r.params, seed=r.seed,
                     hard=hard, gap=gap, nan=not np.isfinite(r.test_ce)))
R = pd.DataFrame(rows)
if R.empty:
    print("no results yet"); raise SystemExit

order = {"D2_comp": 2, "D3_centre": 3, "D4_quad": 4, "D5_quint": 5}
fmt = lambda s: f"{s.mean():6.1f} ± {s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0:4.1f} ({len(s)})"
for ds in sorted(R.dataset.unique(), key=lambda d: order.get(d, 9)):
    print(f"\n=== {ds} (order {order.get(ds, '?')}): % of hard part captured, mean ± s.e. (seeds)")
    sub = R[R.dataset == ds]
    t = sub.groupby(["family", "L"]).hard.apply(fmt).unstack("L")
    print(t.to_string())
    p = sub.groupby(["family", "L"]).params.first().unstack("L")
    print("params:\n" + p.to_string())
    if sub.nan.any():
        print("WARNING: some runs produced NaN loss:", sub[sub.nan][["model", "seed"]].values.tolist())

print("\nTheory check (linear attention-only): a model of depth L can represent interaction order up to 2^L - 1,"
      "\n  i.e. L=1 -> 1, L=2 -> 3, L=3 -> 7. Look for the depth at which each dataset's hard part starts to be captured.")
