#!/bin/bash
#SBATCH --job-name=synthtune
#SBATCH --partition=public
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=synthtune_%A_%a.log

# Step 3a: tuning grid on SELECTION seeds 100-104 (never used for reporting), 7 datasets.
#   12 self-attention configs + 10 quadratic-DeepHalo configs, longer patience (10, 6).
# Results go to a separate folder: synth_tune/ . Resumable: re-run  sbatch run_tune.sbatch
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f box_calib.json ] || { echo "box_calib.json missing"; exit 1; }

OUT=synth_tune
DATA=D0_lcl,D1_sim,D2_comp,D3_centre,D1b_sim,D2b_comp,D3b_centre
SA=""
for L in 1 2 3; do for W in 32 64; do for H in 2 4; do SA="$SA,sa_L${L}_w${W}_h${H}"; done; done; done
DHQ=""
for L in 2 4 5; do for E in 16 32 64; do DHQ="$DHQ,dhq_L${L}_e${E}_H8"; done; done
DHQ="$DHQ,dhq_L4_e32_H16"
MODELS="${SA:1}${DHQ}"
echo "models: $MODELS"
PER=5
K=$((PER * 4))
mkdir -p shard_logs_tune
nvidia-smi -L

for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  python synth.py --datasets $DATA --models $MODELS --seeds 100-104 --out $OUT --threads 1 \
      --device cuda --shard $S/$K --hours 14 --patience 10,6 > shard_logs_tune/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
