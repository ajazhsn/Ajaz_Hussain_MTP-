"""
interp_train.py - retrain a few models WITH their weights saved, for interpret.py.

Same data, seeds and training recipe (patience 10, 6) as the final 30-seed run, so the
test losses should match synth_final/ almost exactly (interpret.py checks this).

Usage (called by run_interp.sbatch):  python interp_train.py SHARD NUM_SHARDS [DATASETS] [SEEDS]
   e.g.  python interp_train.py 0 10 D2b_comp,D3b_centre 0-9
"""
import sys, torch, synth

DATA   = ["D0_lcl", "D2_comp", "D3_centre"]
MODELS = ["sa_L3_w32_h2",    # tuned self-attention (3 layers)
          "sa1",             # 1-layer self-attention (width 32, 2 heads): simplest case to read
          "dhq_L4_e16_H8",   # tuned DeepHalo (quadratic)
          "pool_meansd"]     # own features + others' mean and sd (reference for the IIA probe)
SEEDS  = list(range(5))

if __name__ == "__main__":
    i, k = int(sys.argv[1]), int(sys.argv[2])
    data = sys.argv[3].split(",") if len(sys.argv) > 3 else DATA
    seeds = synth.parse_seeds(sys.argv[4]) if len(sys.argv) > 4 else SEEDS
    torch.set_num_threads(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev} shard {i}/{k}", flush=True)
    synth.run_jobs(data, MODELS, seeds, "synth_interp", dev, shard=(i, k), save_weights=True,
                   hours=5, phases=((1e-3, 10), (1e-4, 6)))
