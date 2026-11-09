"""
size_summary.py - results of the training-size sweep (step 4).
Usage:  python size_summary.py            (reads synth_size/n*/ and synth_final/ for 285k)

Per dataset and training size:
  beyond mean+sd % = (pool_meansd - model) / (pool_meansd - oracle)   [hard part captured]
  gap to twin %    = (twin - model) / (twin - oracle)                   [all context captured]
  wins vs mean+sd  = seeds where the model's test loss is below pool_meansd's (paired)
"""
import glob, json, os
import numpy as np, pandas as pd

rows = []
for d in sorted(glob.glob("synth_size/n*")) + ["synth_final"]:
    rd = f"{d}/results"
    if not os.path.isdir(rd): continue
    for f in os.listdir(rd):
        if f.endswith(".json"):
            r = json.load(open(f"{rd}/{f}")); r["n_train"] = r.get("n_train", 285000); rows.append(r)
df = pd.DataFrame(rows)
df = df[df.dataset.isin(["D0_lcl", "D2_comp", "D3_centre", "D2b_comp", "D3b_centre"])]
orc = df[df.model == "oracle"].groupby(["dataset", "seed"]).test_ce.first().rename("oracle")
m = df[df.model != "oracle"].join(orc, on=["dataset", "seed"])
key = ["dataset", "n_train", "seed"]
ref = m[m.model == "pool_meansd"].set_index(key).test_ce.rename("pm")
tw = m[m.model == "sa1_self"].set_index(key).test_ce.rename("twin")
m = m.join(ref, on=key).join(tw, on=key)
m["beyond"] = 100 * (m.pm - m.test_ce) / (m.pm - m.oracle)
m["gap_twin"] = 100 * (m.twin - m.test_ce) / (m.twin - m.oracle)
m["win_pm"] = m.test_ce < m.pm

def ms(x):
    x = x.dropna()
    return f"{x.mean():6.1f}±{x.std(ddof=1) / np.sqrt(len(x)):4.1f}" if len(x) > 1 else "   n/a   "

for ds in ["D0_lcl", "D2_comp", "D3_centre", "D2b_comp", "D3b_centre"]:
    d = m[m.dataset == ds]
    if d.empty: continue
    metric = "gap_twin" if ds == "D0_lcl" else "beyond"
    name = "% of context captured (gap to twin closed)" if ds == "D0_lcl" else "% of hard part captured (beyond mean+sd)"
    print(f"\n=== {ds}: {name}; [wins vs mean+sd]")
    print(f"{'n_train':>9s} " + " ".join(f"{k:>24s}" for k in ["sa_L3_w32_h2", "dhq_L4_e16_H8", "pool_meansd"]))
    for n, g in d.groupby("n_train"):
        cells = []
        for k in ["sa_L3_w32_h2", "dhq_L4_e16_H8", "pool_meansd"]:
            e = g[g.model == k]
            if e.empty: cells.append(f"{'-':>24s}"); continue
            w = "" if k == "pool_meansd" else f" [{int(e.win_pm.sum())}/{len(e)}]"
            cells.append(f"{ms(e[metric]) + w:>24s}")
        print(f"{n:9d} " + " ".join(cells))
