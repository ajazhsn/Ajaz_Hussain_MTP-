#!/bin/bash
#SBATCH --job-name=synthbox
#SBATCH --partition=public
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=synthbox_%A_%a.log

# Step 2b: box-kernel robustness, D1b-D3b x 30 seeds. Needs box_calib.json (from calib_box.py --pick).
# Your QOS allows 2 GPUs at once, so 2 tasks run and 2 wait - that's expected.
# Resumable: re-run  sbatch run_box.sbatch  after a timeout; finished runs are skipped.

source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f box_calib.json ] || { echo "box_calib.json missing - run: python calib_box.py --pick <radius>"; exit 1; }
OUT=synth_out
DATA=D1b_sim,D2b_comp,D3b_centre
HEAD=sa1_self,pool_meansd,sa1,sa2,dhq4   # twin, mean+sd, attention, paper DeepHalo
REST=pool_mean,dhq_s,dh4                  # extras
PER=5
K=$((PER * 4))
mkdir -p shard_logs_box
nvidia-smi -L

for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  (
    python synth.py --datasets $DATA --models $HEAD --out $OUT --threads 1 --device cuda --shard $S/$K --hours 6
    python synth.py --datasets $DATA --models $REST --out $OUT --threads 1 --device cuda --shard $S/$K --hours 6
  ) > shard_logs_box/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
