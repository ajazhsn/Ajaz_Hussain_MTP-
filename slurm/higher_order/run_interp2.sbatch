#!/bin/bash
#SBATCH --job-name=interp
#SBATCH --partition=public
#SBATCH --array=0-1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=interp_%A_%a.log

# Interpretation, step 1: retrain 4 models x 3 datasets (D0, D2, D3) x 5 seeds WITH weights saved.
#   Same seeds and recipe as synth_final, so test losses should match it.
#   Results + weights: synth_interp/ . Resumable: re-run  sbatch run_interp.sbatch
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

PER=5
K=$((PER * 2))
mkdir -p shard_logs_interp
nvidia-smi -L
for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  python interp_train.py $S $K > shard_logs_interp/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
