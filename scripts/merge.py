"""
merge.py: collect every finished task into one CSV per stage, and (optionally) report on the pilot.

  python scripts/merge.py                     # writes results/merged/<stage>.csv for every stage that has results
  python scripts/merge.py --pilot-report      # checks the pilot and projects how long each tier will take
  python scripts/merge.py --pilot-report --gpus 4 --workers-per-gpu 3
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))   # run from the repo root without installing
import os, json, argparse
import pandas as pd
from lcl.tasks import stage_tasks, task_id, STAGES

OUT = os.environ.get("LCL_OUT", "results")


def load_stage(stage):
    d = os.path.join(OUT, stage)
    rows, tasks = [], []
    if not os.path.isdir(d): return pd.DataFrame(), pd.DataFrame()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"): continue
        with open(os.path.join(d, fn)) as f: r = json.load(f)
        for row in r["rows"]:
            row = dict(row); row["task_id"] = r["task_id"]; row["task_seconds"] = r["task_seconds"]
            row.setdefault("exp", r["task"]["exp"])
            rows.append(row)
        tasks.append(dict(task_id=r["task_id"], exp=r["task"]["exp"], seconds=r["task_seconds"],
                          key=json.dumps({k: v for k, v in r["task"].items() if k not in ("seed", "init_seed", "priority")},
                                         sort_keys=True)))
    return pd.DataFrame(rows), pd.DataFrame(tasks)


def merge_all():
    os.makedirs(os.path.join(OUT, "merged"), exist_ok=True)
    for st in STAGES:
        df, _ = load_stage(st)
        if df.empty: continue
        path = os.path.join(OUT, "merged", f"{st}.csv")
        df.to_csv(path, index=False)
        errs = [f for f in os.listdir(os.path.join(OUT, st)) if f.endswith(".error.txt")]
        print(f"{st:<6} {len(df):>6} rows -> {path}" + (f"   ({len(errs)} FAILED tasks, see *.error.txt)" if errs else ""))


def pilot_report(gpus, wpg):
    df, tk = load_stage("pilot")
    if df.empty: print("no pilot results yet"); return
    lad = df[df.exp == "ladder"]
    print("=== 1. probe checks at every number of features (V1/V2 must be ~1e-15; V4 fidelity < 1; V0 size 0) ===")
    print(lad[["d", "rung", "fro_hat", "cosine", "fid_A", "max_err_A", "max_err_theta", "c_slope", "c_ratio"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    tr = df[df.exp != "ladder"].copy()
    tr["budget_hit"] = tr.budget_hit.fillna(False).astype(bool)
    print("\n=== 2. training sanity ===")
    print(f"model trainings: {len(tr)} | hit the epoch budget: {int(tr.budget_hit.sum())}")
    if tr.budget_hit.any():
        print("epoch-budget hits per experiment and model:")
        print(tr[tr.budget_hit].groupby(["exp", "model"]).size().to_string())
    errs = [f for f in os.listdir(os.path.join(OUT, "pilot")) if f.endswith(".error.txt")]
    print(f"failed pilot tasks: {len(errs)}" + (" -> open results/pilot/*.error.txt" if errs else ""))
    print("\nslowest single model trainings (seconds):")
    print(tr.nlargest(8, "train_seconds")[["exp", "model", "d", "n", "train_seconds", "epochs"]].to_string(index=False))

    print("\n=== 3. projected time per tier (from pilot timings) ===")
    sec = dict(zip(tk.key, tk.seconds))
    for st in ["tune", "tier1", "tier2", "tier3"]:
        total, missing = 0.0, 0
        for t in stage_tasks(st):
            key = json.dumps({k: v for k, v in t.items() if k not in ("seed", "init_seed", "priority")}, sort_keys=True)
            if st == "tune":
                total += 0.0; continue
            if key in sec: total += sec[key]
            else: missing += 1
        if st == "tune":
            print(f"tune  : {len(stage_tasks(st))} short tasks (no probe)"); continue
        par = gpus * wpg
        print(f"{st:<6}: {total / 3600:6.1f} GPU-hours in sequence  ->  ~{total / 3600 / par:5.1f} h with "
              f"{gpus} GPUs x {wpg} workers" + (f"  ({missing} tasks without a pilot timing)" if missing else ""))
    print("\n(running several workers on one GPU is not perfectly parallel; treat these as lower bounds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-report", action="store_true")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--workers-per-gpu", type=int, default=3)
    a = ap.parse_args()
    merge_all()
    if a.pilot_report:
        print(); pilot_report(a.gpus, a.workers_per_gpu)
