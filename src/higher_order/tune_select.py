"""
paired.py - paired comparisons between models, seed by seed.
Usage:  python paired.py synth_out          (needs numpy + pandas)

For each pair (A, B) and dataset: diff = test_ce(A) - test_ce(B) on the SAME seed
(same data, same truth), so seed-to-seed noise cancels.
  mean diff < 0  -> A is better (lower loss than B)
  t              -> mean / standard error  (|t| > ~2.8 is clear with 30 seeds)
  A wins         -> on how many seeds A had the lower loss
Pairs whose runs don't exist yet are skipped.
"""
import json, os, sys
import numpy as np, pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "synth_out"
rd = f"{out}/results"
df = pd.DataFrame([json.load(open(f"{rd}/{f}")) for f in os.listdir(rd) if f.endswith(".json")])
ce = df[df.model != "oracle"].pivot_table(index=["dataset", "seed"], columns="model", values="test_ce")

PAIRS = [
    # main comparisons
    ("sa1", "pool_meansd", "1-layer attention vs mean+sd"),
    ("sa2", "pool_meansd", "2-layer attention vs mean+sd"),
    ("dh4", "pool_meansd", "DeepHalo (paper) vs mean+sd"),
    ("sa1", "dh4",         "1-layer attention vs DeepHalo (paper)"),
    ("sa2", "dh4",         "2-layer attention vs DeepHalo (paper)"),
    ("sa2", "sa1",         "2 layers vs 1 layer"),
    ("dh4", "dh1",         "DeepHalo 4 layers vs 1 layer"),
    # size controls (appear once the control runs exist)
    ("sa1_wide", "sa1",    "1 layer: wider vs normal"),
    ("sa2", "sa1_wide",    "depth vs width at equal size (~17k)"),
    ("sa2_small", "sa1",   "depth vs width at equal size (~9k)"),
    ("sa1", "dh_s",        "attention vs DeepHalo at equal size (~9k)"),
    ("dh_big", "dh4",      "DeepHalo 4x wider vs paper size"),
    ("sa2", "dh_big",      "2-layer attention vs big DeepHalo"),
    # quadratic (paper) DeepHalo
    ("dhq4", "dh4",        "DeepHalo quadratic vs linear (paper size)"),
    ("dhq4", "pool_meansd","DeepHalo quadratic vs mean+sd"),
    ("dhq4", "dhq2",       "DeepHalo quadratic 4 vs 2 layers"),
    ("sa1", "dhq4",        "1-layer attention vs DeepHalo quadratic"),
    ("sa2", "dhq4",        "2-layer attention vs DeepHalo quadratic"),
    ("sa1", "dhq_s",       "attention vs DeepHalo-Q at equal size (~9k)"),
    # tuned configurations (synth_final)
    ("sa_L3_w32_h2", "pool_meansd",   "TUNED attention vs mean+sd"),
    ("dhq_L4_e16_H8", "pool_meansd",  "TUNED DeepHalo-Q vs mean+sd"),
    ("sa_L3_w32_h2", "dhq_L4_e16_H8", "TUNED attention vs TUNED DeepHalo-Q"),
    ("sa_L3_w32_h2", "sa1_self",      "TUNED attention vs blindfolded twin"),
]

for ds in ["D0_lcl", "D1_sim", "D2_comp", "D3_centre", "D1b_sim", "D2b_comp", "D3b_centre"]:
    if ds not in ce.index.get_level_values(0): continue
    d = ce.loc[ds]
    print(f"\n=== {ds}")
    print(f"{'comparison':40s} {'A':>10s} {'B':>12s} {'n':>3s} {'mean diff A-B':>15s} {'s.e.':>8s} {'t':>7s} {'A wins':>8s}")
    for a, b, label in PAIRS:
        if a not in d or b not in d: continue
        x = (d[a] - d[b]).dropna()
        if len(x) < 2: continue
        se = x.std(ddof=1) / np.sqrt(len(x))
        print(f"{label:40s} {a:>10s} {b:>12s} {len(x):3d} {x.mean():15.5f} {se:8.5f} {x.mean() / se:7.1f} "
              f"{int((x < 0).sum()):>4d}/{len(x)}")
