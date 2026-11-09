"""
summarize.py - read synth_out/results/*.json and print the main tables.
Usage:  python summarize.py synth_out          (needs only numpy + pandas)

Metrics (all on the test set, all measured against the TRUE choice probabilities):
  excess   = model test_ce - oracle test_ce      (>= 0; nats lost versus knowing the true formula)
  ctx_rec  = (twin - model) / (twin - oracle)    (% of the context gap closed; twin = sa1_self, the
                                                 blindfolded transformer, i.e. the best no-context model)
  hard_rec = (pool_meansd - model) / (pool_meansd - oracle)
                                                 (% of what mean+sd cannot explain that the model finds)
Mean +- standard error over seeds; only seeds where the needed runs exist are used.
"""
import json, os, sys
import numpy as np, pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "synth_out"
rd = f"{out}/results"
df = pd.DataFrame([json.load(open(f"{rd}/{f}")) for f in os.listdir(rd) if f.endswith(".json")])
orc = df[df.model == "oracle"].set_index(["dataset", "seed"])[["test_ce", "plain_test_ce"]]
orc.columns = ["oracle", "plain"]
m = df[df.model != "oracle"].join(orc, on=["dataset", "seed"])
pm = m[m.model == "pool_meansd"].set_index(["dataset", "seed"]).test_ce.rename("pm")
tw = m[m.model == "sa1_self"].set_index(["dataset", "seed"]).test_ce.rename("twin")
m = m.join(pm, on=["dataset", "seed"]).join(tw, on=["dataset", "seed"])
m["excess"] = m.test_ce - m.oracle
m["ctx_rec"] = 100 * (m.twin - m.test_ce) / (m.twin - m.oracle)
m["hard_rec"] = 100 * (m.pm - m.test_ce) / (m.pm - m.oracle)

def ms(x, nd=3):
    x = x.dropna()
    if len(x) == 0: return "       n/a"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return f"{x.mean():8.{nd}f} +- {se:7.{nd}f}"

order = ["mlp", "sa1_self", "pool_mean", "pool_meansd", "sa1", "sa1_wide", "sa2_small", "sa2", "dh_s", "dh1", "dh2", "dh3", "dh4", "dh_big", "dhq_s", "dhq2", "dhq4"]
order += sorted(set(m.model) - set(order))          # any other models (e.g. tuned configs) at the end
for ds in ["D0_lcl", "D1_sim", "D2_comp", "D3_centre", "D1b_sim", "D2b_comp", "D3b_centre"]:
    d = m[m.dataset == ds]
    if d.empty:
        print(f"\n{ds}: no results yet"); continue
    o = orc.loc[ds]
    print(f"\n=== {ds}   seeds with oracle: {len(o)}   oracle CE {o.oracle.mean():.4f}   "
          f"theta-only CE {o.plain.mean():.4f}")
    print(f"{'model':12s} {'n':>3s} {'params':>7s} {'excess (nats)':>22s} {'gap to twin closed %':>24s} {'beyond mean+sd %':>24s}")
    for k in order:
        e = d[d.model == k]
        if e.empty: continue
        hard = ("(reference)" if k == "pool_meansd" else "n/a (lcl)" if ds == "D0_lcl" else ms(e.hard_rec))
        print(f"{k:12s} {len(e):3d} {int(e.params.iloc[0]):7d} {ms(e.excess, 4):>22s} {ms(e.ctx_rec):>24s} {hard:>24s}")
