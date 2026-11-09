#!/bin/bash
#SBATCH --job-name=interp2
#SBATCH --partition=public
#SBATCH --array=0-1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=interp2_%A_%a.log

# Interpretation, extension: add the box-kernel datasets (D2b, D3b) and seeds 5-9 everywhere,
#   so every dataset has 10 seeds. Finished runs (seeds 0-4 of D0/D2/D3) are skipped automatically.
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f box_calib.json ] || { echo "box_calib.json missing"; exit 1; }

DATA=D2b_comp,D3b_centre,D0_lcl,D2_comp,D3_centre
PER=5
K=$((PER * 2))
mkdir -p shard_logs_interp2
for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  python interp_train.py $S $K $DATA 0-9 > shard_logs_interp2/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
