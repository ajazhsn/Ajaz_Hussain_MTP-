"""
calib_box.py - calibration for the box-kernel robustness datasets (no training).

Step A (sweep):   python calib_box.py
   For each radius r and each of D1b/D2b/D3b: find lambda so that the part NOT explained
   by own + others' mean & sd is 0.03 nats (same rule as the Gaussian datasets), and report
   the low-order share and how often the effect is non-zero. Writes calib_box_table.csv.
Step B (pick):    python calib_box.py --pick 0.3
   Writes box_calib.json for the chosen radius; synth.py reads it automatically.
"""
import argparse, json, os
import numpy as np, pandas as pd
import synth

DS = ["D1b_sim", "D2b_comp", "D3b_centre"]
TARGET_HARD = 0.03

ap = argparse.ArgumentParser()
ap.add_argument("--radii", default="0.2,0.25,0.3,0.4")
ap.add_argument("--pick", type=float, default=None)
a = ap.parse_args()

if a.pick is not None:
    t = pd.read_csv("calib_box_table.csv")
    t = t[np.isclose(t.radius, a.pick)]
    assert len(t) == 3, f"radius {a.pick} not in calib_box_table.csv"
    out = {"radius": a.pick, "lambda": {r.dataset: round(float(r.lam), 4) for r in t.itertuples()}}
    json.dump(out, open("box_calib.json", "w"), indent=1)
    print("wrote box_calib.json:", out)
    raise SystemExit

lams = list(np.geomspace(0.01, 20, 16))
rows = []
for r in map(float, a.radii.split(",")):
    for ds in DS:
        synth.DATASETS[ds]["sigma"] = r
        X = np.random.default_rng(7).normal(size=(5000, synth.N_ITEMS, synth.D))
        c = synth.context_raw(X, ds, None)
        nonzero = float((c > 0).mean())
        g = pd.DataFrame(synth.calibrate(ds, lams)).groupby("lam").mean(numeric_only=True).reset_index()
        g = g[g.gap_low > 0]
        lg = np.log(g.gap_low.values)
        lam = float(np.interp(np.log(TARGET_HARD), lg, g.lam))
        twin = float(np.interp(np.log(TARGET_HARD), lg, g.gap_twin))
        edge = lam <= g.lam.min() * 1.01 or lam >= g.lam.max() * 0.99
        rows.append(dict(dataset=ds, radius=r, lam=lam, twin_gap=twin,
                         low_share=100 * (1 - TARGET_HARD / twin), frac_items_affected=nonzero,
                         r2_low=float(g.r2_low.iloc[0]), at_grid_edge=edge))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

t = pd.DataFrame(rows); t.to_csv("calib_box_table.csv", index=False)
print("\n", t.round(4).to_string(index=False))
print("\nGaussian datasets for comparison (sigma 0.2): low_share D1 ~3%, D2 ~30%, D3 ~61%")
