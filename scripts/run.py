"""
run.py: the worker. Start as many copies as you like (several per GPU is fine: the models are small).
Every copy walks through the task list and takes the next task nobody has taken yet, so copies never
duplicate work, and a stopped job can simply be resubmitted: finished tasks are skipped.

  python scripts/run.py --stage pilot,tune            # run these stages, in this order
  python scripts/run.py --finalize-tuning             # pick lr / weight decay per model from the tune stage
  python scripts/run.py --status                      # how many tasks are done / running / failed per stage
  python scripts/run.py --stage tier1 --retry-errors  # also retry tasks that failed before
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))   # run from the repo root without installing
import os, sys, json, time, socket, argparse, traceback
import numpy as np

from lcl.core import run_task, VERSION, DEVICE, QUICK, HP_DEFAULT
from lcl.tasks import stage_tasks, task_id, STAGES

OUT = os.environ.get("LCL_OUT", "results")
HP_FILE = os.path.join(OUT, "hp.json")


def paths(stage, tid):
    d = os.path.join(OUT, stage)
    return (os.path.join(d, tid + ".json"), os.path.join(d, tid + ".claim"), os.path.join(d, tid + ".error.txt"))


def try_claim(claim, stale_hours):
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', '-')} {time.time()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(claim) > stale_hours * 3600:   # the worker that claimed it died
                os.remove(claim)
                return try_claim(claim, stale_hours)
        except FileNotFoundError:
            return try_claim(claim, stale_hours)
        return False


def load_hp(stage):
    if os.path.exists(HP_FILE):
        with open(HP_FILE) as f: return json.load(f)
    if stage in ("pilot", "tune"):
        return {}
    sys.exit(f"{HP_FILE} not found: run the tune stage and then  python scripts/run.py --finalize-tuning  first")


def to_jsonable(v):
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, np.bool_): return bool(v)
    return v


def run_stage(stage, stale_hours, retry_errors, only=""):
    os.makedirs(os.path.join(OUT, stage), exist_ok=True)
    hp = load_hp(stage)
    tasks = stage_tasks(stage)
    if only:                                          # e.g. --only E3_rank,E4_sparsity  (for CPU workers)
        prefixes = tuple(p for p in only.split(",") if p)
        tasks = [t for t in tasks if t["exp"].startswith(prefixes)]
    done_here = 0
    print(f"[{socket.gethostname()} pid {os.getpid()}] stage {stage}: {len(tasks)} tasks, device={DEVICE}, quick={QUICK}", flush=True)
    for task in tasks:
        tid = task_id(task)
        res, claim, err = paths(stage, tid)
        if os.path.exists(res): continue
        if os.path.exists(err) and not retry_errors: continue
        if not try_claim(claim, stale_hours): continue
        t0 = time.time()
        try:
            rows = run_task(task, hp)
            out = dict(task=task, task_id=tid, stage=stage, version=VERSION, host=socket.gethostname(),
                       task_seconds=time.time() - t0,
                       rows=[{k: to_jsonable(v) for k, v in r.items()} for r in rows])
            tmp = res + f".tmp{os.getpid()}"
            with open(tmp, "w") as f: json.dump(out, f)
            os.replace(tmp, res)                        # atomic: a result file is always complete
            if os.path.exists(err): os.remove(err)
            done_here += 1
            print(f"  done {tid}  ({time.time() - t0:.0f}s)", flush=True)
        except Exception:
            with open(err, "w") as f: f.write(json.dumps(task) + "\n\n" + traceback.format_exc())
            print(f"  FAILED {tid}: see {err}", flush=True)
        finally:
            if os.path.exists(claim): os.remove(claim)
    print(f"stage {stage}: this worker finished {done_here} tasks", flush=True)


def status():
    for stage in STAGES:
        tasks = stage_tasks(stage)
        n_done = n_run = n_err = 0
        for t in tasks:
            res, claim, err = paths(stage, task_id(t))
            n_done += os.path.exists(res); n_run += os.path.exists(claim); n_err += os.path.exists(err)
        print(f"{stage:<6} {n_done:>5} done | {n_run:>3} running | {n_err:>3} failed | {len(tasks) - n_done:>5} left of {len(tasks)}")


def finalize_tuning():
    rows = []
    d = os.path.join(OUT, "tune")
    for fn in os.listdir(d) if os.path.isdir(d) else []:
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f: rows += json.load(f)["rows"]
    if not rows: sys.exit("no tuning results yet")
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(["family", "lr", "wd"]).agg(val_nll=("val_nll", "mean"), runs=("val_nll", "size"),
                                                budget_hits=("budget_hit", "sum")).reset_index()
    hp, notes = {}, []
    lrs = sorted(df.lr.unique())
    for fam, gg in g.groupby("family"):
        best = gg.loc[gg.val_nll.idxmin()]
        hp[fam] = dict(lr=float(best.lr), wd=float(best.wd))
        spread = gg.val_nll.max() - gg.val_nll.min()
        edge = " (edge of grid)" if best.lr in (lrs[0], lrs[-1]) else ""
        notes.append(f"{fam:<10} lr={best.lr:<7g} wd={best.wd:<7g} val_nll={best.val_nll:.5f}  "
                     f"spread over grid={spread:.5f}  runs={int(gg.runs.sum())}  budget hits={int(gg.budget_hits.sum())}{edge}")
    with open(HP_FILE, "w") as f: json.dump(hp, f, indent=2)
    print("\n".join(notes))
    print(f"\nsaved {HP_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="")
    ap.add_argument("--stale-hours", type=float, default=12.0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--finalize-tuning", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated experiment name prefixes to restrict to")
    a = ap.parse_args()
    if a.status: status()
    elif a.finalize_tuning: finalize_tuning()
    else:
        for st in [s for s in a.stage.split(",") if s]:
            run_stage(st, a.stale_hours, a.retry_errors, a.only)
