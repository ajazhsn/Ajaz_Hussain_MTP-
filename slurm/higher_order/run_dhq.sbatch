#!/bin/bash
#SBATCH --job-name=synthsize
#SBATCH --partition=public
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=synthsize_%A_%a.log

# Step 4: training-size sweep with the TUNED models (+ mean+sd reference and twin).
#   Sizes 11k, 30k, 100k (30 seeds) and 1M (10 seeds); 285k already exists in synth_final/.
#   Val/test sets are identical at every size and training sets are nested.
#   Results: synth_size/n<N>/ . Resumable: re-run  sbatch run_size.sbatch
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f box_calib.json ] || { echo "box_calib.json missing"; exit 1; }

DATA=D0_lcl,D2_comp,D3_centre,D2b_comp,D3b_centre
MODELS=sa_L3_w32_h2,dhq_L4_e16_H8,pool_meansd,sa1_self
PER=5
K=$((PER * 4))
mkdir -p shard_logs_size
nvidia-smi -L

for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  (
    # small data: same patience in epochs, but allow more epochs (each epoch is much shorter)
    for NE in "11000 2000" "30000 800" "100000 240"; do
      set -- $NE
      python synth.py --datasets $DATA --models $MODELS --seeds 0-29 --n-train $1 --max-epochs $2 \
        --out synth_size/n$1 --threads 1 --device cuda --shard $S/$K --hours 8 --patience 10,6
    done
    python synth.py --datasets $DATA --models $MODELS --seeds 0-9 --n-train 1000000 --max-epochs 80 \
      --out synth_size/n1000000 --threads 1 --device cuda --shard $S/$K --hours 10 --patience 10,6
  ) > shard_logs_size/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
