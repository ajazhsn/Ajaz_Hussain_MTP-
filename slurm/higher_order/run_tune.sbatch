#!/bin/bash
#SBATCH --job-name=synthctrl
#SBATCH --partition=public
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=synthctrl_%A_%a.log

# Size controls on D1-D3 x 30 seeds:
#   sa1_wide  (1 layer, ~17.9k = sa2's size)    sa2_small (2 layers, ~8.4k = sa1's size)
#   dh_s      (DeepHalo, ~8.9k = sa1's size)    dh_big    (DeepHalo 4x wider, ~161k)
# Your QOS allows 2 GPUs at once, so 2 tasks run and 2 wait - that's expected.
# Resumable: re-run  sbatch run_ctrl.sbatch  after a timeout; finished runs are skipped.

source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=synth_out
DATA=D1_sim,D2_comp,D3_centre
HEAD=sa1_wide,sa2_small,dh_s      # the size-matched controls first
REST=dh_big                       # the big DeepHalo last (slowest)
PER=5
K=$((PER * 4))
mkdir -p shard_logs_ctrl
nvidia-smi -L

for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  (
    python synth.py --datasets $DATA --models $HEAD --out $OUT --threads 1 --device cuda --shard $S/$K --hours 6
    python synth.py --datasets $DATA --models $REST --out $OUT --threads 1 --device cuda --shard $S/$K --hours 6
  ) > shard_logs_ctrl/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
