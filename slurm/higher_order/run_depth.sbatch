#!/bin/bash
#SBATCH --job-name=synthdepth
#SBATCH --partition=public
#SBATCH --array=0-1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=synthdepth_%A_%a.log

# Depth-order study, step 2: depth L = 1..4 at a FIXED parameter budget, three families:
#   full self-attention (~26k params), attention-only softmax and attention-only linear (~16k params),
#   plus the mean+sd pool and the blindfolded twin as references.
#   Datasets of order 2, 3, 4, 5; 5 seeds; 285k training slates. Results: synth_depth/  (resumable)
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f order_calib.json ] || { echo "order_calib.json missing - run calib_order.sbatch first"; exit 1; }

DATA=D2_comp,D3_centre,D4_quad,D5_quint
FULL=sa_L1_w56_h2,sa_L2_w40_h2,sa_L3_w32_h2,sa_L4_w28_h2
SOFT=aosoft_L1_w56_h2,aosoft_L2_w42_h2,aosoft_L3_w34_h2,aosoft_L4_w30_h2
LIN=aolin_L1_w56_h2,aolin_L2_w42_h2,aolin_L3_w34_h2,aolin_L4_w30_h2
MODELS=pool_meansd,sa1_self,$FULL,$SOFT,$LIN
PER=5
K=$((PER * 2))
mkdir -p shard_logs_depth
nvidia-smi -L
for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  python synth.py --datasets $DATA --models $MODELS --seeds 0-4 --out synth_depth --threads 1 \
      --device cuda --shard $S/$K --hours 19 --patience 10,6 > shard_logs_depth/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
