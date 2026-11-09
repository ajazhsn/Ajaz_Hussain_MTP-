"""
calib_order.py - calibration of the order-4 and order-5 datasets (no training, CPU only).

Same rule as every other higher-order dataset: choose lambda so that the part of the effect
NOT explained by own features + others' mean & sd ("hard part") is 0.03 nats.
Writes order_calib.json; synth.py reads it automatically.

Usage:  python calib_order.py
"""
import json
import numpy as np, pandas as pd
import synth

DS = ["D4_quad", "D5_quint"]
TARGET_HARD = 0.03
lams = list(np.geomspace(0.005, 2.0, 14))

rows, out = [], {}
for ds in DS:
    print("calibrating", ds, flush=True)
    g = pd.DataFrame(synth.calibrate(ds, lams)).groupby("lam").mean(numeric_only=True).reset_index()
    print(g[["lam", "gap_twin", "gap_low", "r2_own", "r2_low"]].round(4).to_string(index=False), flush=True)
    g = g[g.gap_low > 0]
    lg = np.log(g.gap_low.values)
    lam = float(np.exp(np.interp(np.log(TARGET_HARD), lg, np.log(g.lam.values))))
    twin = float(np.exp(np.interp(np.log(TARGET_HARD), lg, np.log(g.gap_twin.values))))
    edge = lam <= g.lam.min() * 1.01 or lam >= g.lam.max() * 0.99
    rows.append(dict(dataset=ds, lam=round(lam, 4), twin_gap=round(twin, 4),
                     low_share_pct=round(100 * (1 - TARGET_HARD / twin), 1), at_grid_edge=edge))
    out[ds] = round(lam, 4)
    print("  ->", rows[-1], flush=True)

json.dump({"lambda": out, "target_hard": TARGET_HARD}, open("order_calib.json", "w"), indent=1)
print("\n", pd.DataFrame(rows).to_string(index=False))
print("\nwrote order_calib.json:", out)
print("for comparison (Gaussian, sigma 0.2): low-order share D2 ~30%, D3 ~61%")
print("if at_grid_edge is True, widen the lambda grid and rerun")
