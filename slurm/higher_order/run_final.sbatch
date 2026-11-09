#!/bin/bash
#SBATCH --job-name=synthdhq
#SBATCH --partition=public
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=synthdhq_%A_%a.log

# Paper's quadratic (Hadamard) DeepHalo on D0-D3 x 30 seeds: dhq_s (~9.4k), dhq2 (~24k), dhq4 (~44k).
# Your QOS allows 2 GPUs at once, so 2 tasks run and 2 wait - that's expected.
# Resumable: re-run  sbatch run_dhq.sbatch  after a timeout; finished runs are skipped.

source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=synth_out
DATA=D0_lcl,D1_sim,D2_comp,D3_centre
HEAD=dhq_s,dhq2              # quadratic DeepHalo: equal size, 2 layers
REST=dhq4                     # quadratic DeepHalo, paper config
PER=5
K=$((PER * 4))
mkdir -p shard_logs_dhq
nvidia-smi -L

for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  (
    python synth.py --datasets $DATA --models $HEAD --out $OUT --threads 1 --device cuda --shard $S/$K --hours 6
    python synth.py --datasets $DATA --models $REST --out $OUT --threads 1 --device cuda --shard $S/$K --hours 6
  ) > shard_logs_dhq/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
