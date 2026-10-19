"""
curves.py: train each model once on the reference data (16 features, rank 5, 285k sessions, dataset seed 101)
with exactly the protocol of the main runs, and record train / validation loss after EVERY epoch.
The main runs only stored final numbers; this gives the training-curve figures.

  python scripts/curves.py            # worker: trains every model not done yet (start several copies; they share the work)
  python scripts/curves.py --status   # which models are done / running
  python scripts/curves.py --merge    # combine everything into results/curves.json  (upload that file to Colab)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))   # run from the repo root without installing
import os, sys, json, time, copy, socket, argparse
import numpy as np
import torch
import torch.nn.functional as F
import lcl.core as L

OUT = os.path.join(os.environ.get("LCL_OUT", "results"), "curves")
HP_FILE = os.path.join(os.environ.get("LCL_OUT", "results"), "hp.json")
# slowest first, so the long ones start early
MODELS = ["sa_L6", "sa_L4", "sa_L2", "sa_L1", "mp_L6", "mp_L4", "mp_L2",
          "flat_L6", "flat_L4", "flat_L2", "sov_h16", "bil_h16", "oracle"]
D, RANK, SEED = 16, 5, 101
N_TRAIN = 3000 if L.QUICK else 285000


def train_with_history(model, tr, va, lr, wd):
    """Identical to lcl_core.train_model, but records the loss after every epoch."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    hist = dict(train=[], val=[], val_kl=[])
    best, best_state, best_ep = float("inf"), None, 0
    N = len(tr["y"])
    for ep in range(L.MAX_EPOCHS):
        model.train(); opt.zero_grad(); tot = 0.0
        for i in range(0, N, L.CHUNK):
            loss = F.cross_entropy(model(tr["X"][i:i + L.CHUNK]), tr["y"][i:i + L.CHUNK], reduction="sum") / N
            loss.backward(); tot += loss.item()
        opt.step()
        v, vkl = L.eval_split(model, va)
        hist["train"].append(tot); hist["val"].append(v); hist["val_kl"].append(vkl)
        if v < best: best, best_ep, best_state = v, ep, copy.deepcopy(model.state_dict())
        if ep - best_ep >= L.PATIENCE: break
    model.load_state_dict(best_state)
    return hist, best_ep


@torch.no_grad()
def test_metrics(model, s):
    model.eval()
    q = torch.cat([torch.softmax(model(s["X"][i:i + L.CHUNK]), 1) for i in range(0, len(s["y"]), L.CHUNK)])
    pred = q.argmax(1)
    nll, kl = L.eval_split(model, s)
    return dict(test_nll=nll, test_kl=kl,
                accuracy=(pred == s["y"]).float().mean().item(),                    # top item = item chosen
                expected_accuracy=s["p"].gather(1, pred[:, None]).mean().item(),     # same, averaged over the randomness of the choice
                agree_with_truth=(pred == s["p"].argmax(1)).float().mean().item())   # top item = true model's top item


def truth_numbers(dev):
    def nll(s): return float(-torch.log(s["p"].gather(1, s["y"][:, None])).mean())
    p, y = dev["eval_in"]["p"], dev["eval_in"]["y"]
    return dict(floor_train=nll(dev["train"]), floor_val=nll(dev["val"]), test_nll=nll(dev["eval_in"]),
                accuracy=(p.argmax(1) == y).float().mean().item(), expected_accuracy=p.max(1).values.mean().item())


def claim(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', '-')}".encode())
        os.close(fd); return True
    except FileExistsError:
        return False


def work():
    os.makedirs(OUT, exist_ok=True)
    hp = json.load(open(HP_FILE))
    dev = None
    for m in MODELS:
        done, cl = os.path.join(OUT, m + ".json"), os.path.join(OUT, m + ".claim")
        if os.path.exists(done) or not claim(cl): continue
        if dev is None:
            theta, A = L.make_ground_truth(D, RANK, SEED)
            dev = L.build_splits(theta, A, N_TRAIN, D, data_seed=SEED + 1)     # same data as headline seed 101
        h = hp.get(L.family(m), L.HP_DEFAULT)
        torch.manual_seed(SEED)
        model = L.build_model(m, D).to(L.DEVICE)
        t0 = time.time()
        hist, best_ep = train_with_history(model, dev["train"], dev["val"], h["lr"], h["wd"])
        rec = dict(model=m, hist=hist, best_ep=best_ep, lr=h["lr"], wd=h["wd"], params=L.n_params(model),
                   seconds=time.time() - t0, device=L.DEVICE, truth=truth_numbers(dev), **test_metrics(model, dev["eval_in"]))
        tmp = done + ".tmp"; json.dump(rec, open(tmp, "w")); os.replace(tmp, done)
        os.remove(cl)
        print(f"{m:8s} {len(hist['val']):4d} epochs, best at {best_ep + 1:4d}, {rec['seconds']:6.0f} s, "
              f"test NLL {rec['test_nll']:.4f}", flush=True)


def status():
    for m in MODELS:
        st = "done" if os.path.exists(os.path.join(OUT, m + ".json")) else (
             "running" if os.path.exists(os.path.join(OUT, m + ".claim")) else "waiting")
        print(f"{m:8s} {st}")


def merge():
    out = {}
    for m in MODELS:
        p = os.path.join(OUT, m + ".json")
        if os.path.exists(p): out[m] = json.load(open(p))
    if not out: sys.exit("no finished models yet")
    out["_truth"] = next(iter(out.values()))["truth"]
    path = os.path.join(os.path.dirname(OUT), "curves.json")
    json.dump(out, open(path, "w"))
    print(f"{len(out) - 1} of {len(MODELS)} models -> {path}")
    missing = [m for m in MODELS if m not in out]
    if missing: print("still missing:", ", ".join(missing))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true"); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    status() if a.status else (merge() if a.merge else work())
