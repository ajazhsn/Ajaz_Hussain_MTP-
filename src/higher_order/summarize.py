"""
tune_select.py - pick ONE configuration per model family from the tuning grid.
Usage:  python tune_select.py synth_tune

Rule: for every (dataset, seed), regret = val NLL - best val NLL in that family on that
(dataset, seed). The chosen config has the lowest regret averaged over ALL datasets and
seeds (validation only - test numbers are shown for information, never used to choose).
"""
import json, os, sys
import numpy as np, pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "synth_tune"
rd = f"{out}/results"
df = pd.DataFrame([json.load(open(f"{rd}/{f}")) for f in os.listdir(rd) if f.endswith(".json")])
orc = df[df.model == "oracle"].set_index(["dataset", "seed"]).test_ce.rename("oracle")
m = df[df.model != "oracle"].join(orc, on=["dataset", "seed"])
m["family"] = np.where(m.model.str.startswith("sa_"), "self-attention", "DeepHalo-quadratic")
m["excess"] = m.test_ce - m.oracle
n_cells = m.groupby(["dataset", "seed"]).ngroups
print(f"runs: {len(m)}   (dataset, seed) cells: {n_cells}   models: {m.model.nunique()}")

for fam, g in m.groupby("family"):
    g = g.copy()
    g["regret"] = g.val_nll - g.groupby(["dataset", "seed"]).val_nll.transform("min")
    t = g.groupby("model").agg(params=("params", "first"), n=("regret", "size"),
                               mean_regret=("regret", "mean"), worst_regret=("regret", "max"))
    t["complete"] = t.n == n_cells
    t = t.sort_values("mean_regret")
    print(f"\n===== {fam}  (lower mean_regret = better; 'complete' = ran on every dataset x seed)")
    print(t.round(5).to_string())
    best = t[t.complete].index[0] if t.complete.any() else t.index[0]
    print(f"\n  -> CHOSEN: {best}")
    # per-dataset view of the top 3: test excess over the oracle (information only)
    top = list(t.index[:3])
    v = g[g.model.isin(top)].pivot_table(index="dataset", columns="model", values="excess", aggfunc="mean")
    print("\n  test excess over oracle (nats), top 3, information only:")
    print(v[top].round(4).to_string())
