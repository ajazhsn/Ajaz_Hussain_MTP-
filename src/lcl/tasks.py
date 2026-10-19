"""
tasks.py: the full experiment design, as a list of tasks per stage.

One task = one setting, one world, one seed (data), all its models trained on the SAME data.
Stages, in the order they should run:
  pilot  : every setting once (1 seed) + probe checks up to 32 features. Checks nothing crashes, measures time.
  tune   : learning rate x weight decay for every model family at the reference setting (validation NLL only).
  tier1  : headline (E1), data size (E2), rank x head size (E3), depth (E5)
  tier2  : sparsity (E4s), number of features (E4d)
  tier3  : hyperparameter robustness (E7), identification (E6)
"""
import json, hashlib
from lcl.core import QUICK

# ------------------------------------------------------------------ reference setting
REF = dict(d=16, rank=5, n=285000, sparsity=0.0)        # 285k = 40k x (16/6)^2: same data per unknown as d=6 at 40k
BOTH = ["no_effect", "context"]
CTX = ["context"]

def seeds(n, start=101):
    return list(range(start, start + (min(n, 2) if QUICK else n)))

def scale_n(n):
    return max(400, n // 100) if QUICK else n

E1_MODELS = ["flat_L2", "flat_L4", "flat_L6", "sa_L1", "sa_L6", "mp_L2", "bil_h16", "sov_h16", "oracle"]
CORE = ["flat_L2", "sa_L1", "mp_L2", "oracle"]
DEPTH_MODELS = ["sa_L1", "sa_L2", "sa_L4", "sa_L6",
                "flat_L2", "flat_L4", "flat_L6", "flatw_L4", "flatw_L6",
                "mp_L2", "mp_L4", "mp_L6", "oracle"]
TUNE_MODELS = DEPTH_MODELS + ["bil_h16", "sov_h16"]
H_GRID = [1, 2, 4, 8, 16]
R_GRID = [1, 2, 4, 8, 16]
FEATURES = [6, 12, 16, 24, 32]


def _task(exp, world, seed, models, d, rank, n, sparsity=0.0, priority=1, **extra):
    t = dict(type="train", exp=exp, d=d, rank=rank, n=scale_n(n), sparsity=sparsity,
             world=world, seed=seed, models=list(models), priority=priority)
    t.update(extra)
    return t


def tier1():
    T = []
    # E1 headline: everything at the reference setting, 30 seeds, both worlds
    for s in seeds(30):
        for w in BOTH:
            T.append(_task("E1_headline", w, s, E1_MODELS, priority=1, **REF))
    # E2 data size at d=16 (same n per unknown as d=6's 2.5k ... 160k, plus 2 larger points)
    for n in [18000, 36000, 71000, 142000, 285000, 570000, 1140000]:
        for s in seeds(15):
            for w in BOTH:
                T.append(_task("E2_datasize", w, s, CORE, d=16, rank=5, n=n, priority=1))
    # E3 rank x head size at d=16 (context only: the floor does not depend on rank)
    for r in R_GRID:
        models = [f"bil_h{h}" for h in H_GRID] + [f"sov_h{h}" for h in H_GRID] + CORE
        for s in seeds(10):
            T.append(_task("E3_rank", "context", s, models, d=16, rank=r, n=285000, priority=1))
    # E5 depth: two data sizes, both worlds
    for n in [71000, 285000]:
        for s in seeds(15):
            for w in BOTH:
                T.append(_task("E5_depth", w, s, DEPTH_MODELS, d=16, rank=5, n=n, priority=1))
    return T


def tier2():
    T = []
    # E4s sparsity, full rank (16), two data sizes, context only (floor does not depend on sparsity)
    for sp in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for n in [71000, 285000]:
            for s in seeds(15):
                T.append(_task("E4_sparsity", "context", s, CORE, d=16, rank=16, n=n, sparsity=sp, priority=2))
    # E4d number of features: data scaled with d^2, and data fixed at 40k
    for d in FEATURES:
        rank = max(1, round(d / 3))
        for arm, n in [("scaled", int(40000 * (d / 6) ** 2)), ("fixed40k", 40000)]:
            if arm == "fixed40k" and d == 6: continue          # same as scaled at d=6
            for s in seeds(10):
                for w in BOTH:
                    T.append(_task("E4_features", w, s, CORE, d=d, rank=rank, n=n, priority=2, arm=arm))
    return T


def tier3():
    T = []
    # E7 hyperparameter robustness: every lr x wd, plus half / double width, at the reference setting
    for lr in [3e-3, 1e-2, 3e-2]:
        for wd in [0.0, 1e-4]:
            for s in seeds(10):
                T.append(_task("E7_hparams", "context", s, ["sa_L1", "mp_L2", "flat_L2"], priority=3,
                               hp_override=dict(lr=lr, wd=wd), **REF))
    for s in seeds(10):
        T.append(_task("E7_width", "context", s,
                       ["sa_L1_x0.5", "sa_L1_x2", "mp_L2_x0.5", "mp_L2_x2", "flat_L2_x0.5", "flat_L2_x2"],
                       priority=3, **REF))
    # E6 identification: same data, different starting weights
    for data_seed in seeds(10, start=201):
        for init in seeds(20, start=1):
            T.append(_task("E6_identification", "context", data_seed, ["sa_L1", "mp_L2", "oracle"],
                           priority=3, init_seed=init, save_A=True, **REF))
    return T


def tune():
    T = []
    for name in TUNE_MODELS:
        for lr in [3e-3, 1e-2, 3e-2]:
            for wd in [0.0, 1e-4]:
                for s in seeds(2, start=1):
                    t = _task("tune", "context", s, [name], priority=0, hp_override=dict(lr=lr, wd=wd), **REF)
                    t["type"] = "tune"
                    T.append(t)
    return T


def pilot():
    T = [dict(type="ladder", exp="ladder", d=d, rank=max(1, round(d / 3)), priority=0) for d in FEATURES]
    seen = set()
    for t in tier1() + tier2() + tier3():
        key = json.dumps({k: v for k, v in t.items() if k not in ("seed", "init_seed")}, sort_keys=True)
        if key in seen: continue
        seen.add(key)
        p = dict(t); p["seed"] = 9001; p["priority"] = 0
        if "init_seed" in p: p["init_seed"] = 1
        T.append(p)
    return T


STAGES = {"pilot": pilot, "tune": tune, "tier1": tier1, "tier2": tier2, "tier3": tier3}


def task_id(task):
    return task["exp"] + "_" + hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()[:14]


def stage_tasks(stage):
    T = STAGES[stage]()
    # long tasks first inside a stage, so short ones fill the gaps at the end
    cost = lambda t: t.get("n", 0) * len(t.get("models", [1])) * (t.get("d", 6) ** 0.5)
    return sorted(T, key=lambda t: (t["priority"], -cost(t)))


if __name__ == "__main__":
    for st in STAGES:
        T = stage_tasks(st)
        n_models = sum(len(t.get("models", [])) for t in T)
        print(f"{st:<6} {len(T):>5} tasks, {n_models:>6} model trainings")
