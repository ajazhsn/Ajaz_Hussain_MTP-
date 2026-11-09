#!/bin/bash
#SBATCH --job-name=synthgpu
#SBATCH --partition=public
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=synthgpu_%A_%a.log

# GPU version of the sweep: 4 array jobs, 1 GPU each, 5 processes sharing that GPU
# (the models are tiny, so one GPU is shared by several runs).
# 4 jobs x 5 processes = 20 shards over all 120 (dataset, seed) pairs.
# Finished runs are skipped, so it is safe on top of earlier results and
# safe to re-run after a crash/timeout:   sbatch run_gpu.sbatch

source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=synth_out
HEAD=sa1,sa1_self,pool_meansd,pool_mean,dh4      # headline comparison first
REST=mlp,sa2,dh1,dh2,dh3                         # extras second
PER=5
K=$((PER * 4))                                   # 20 shards in total
mkdir -p shard_logs_gpu
nvidia-smi -L

for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  (
    python synth.py --models $HEAD --out $OUT --threads 1 --device cuda --shard $S/$K --hours 5
    python synth.py --models $REST --out $OUT --threads 1 --device cuda --shard $S/$K --hours 5
  ) > shard_logs_gpu/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
