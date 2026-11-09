#!/bin/bash
#SBATCH --job-name=synthfinal
#SBATCH --partition=public
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=synthfinal_%A_%a.log

# Step 3b: the two CHOSEN configurations on the 30 REPORTING seeds (0-29), all 7 datasets,
#   plus the mean+sd reference and the blindfolded twin re-trained with the same patience (10, 6).
# Results go to synth_final/ . Resumable: re-run  sbatch run_final.sbatch
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f box_calib.json ] || { echo "box_calib.json missing"; exit 1; }

OUT=synth_final
DATA=D0_lcl,D1_sim,D2_comp,D3_centre,D1b_sim,D2b_comp,D3b_centre
MODELS=sa_L3_w32_h2,dhq_L4_e16_H8,pool_meansd,sa1_self
echo "models: $MODELS"
PER=5
K=$((PER * 4))
mkdir -p shard_logs_final
nvidia-smi -L

for p in $(seq 0 $((PER - 1))); do
  S=$((SLURM_ARRAY_TASK_ID * PER + p))
  python synth.py --datasets $DATA --models $MODELS --seeds 0-29 --out $OUT --threads 1 \
      --device cuda --shard $S/$K --hours 14 --patience 10,6 > shard_logs_final/shard_${S}.log 2>&1 &
done
wait
echo "array task $SLURM_ARRAY_TASK_ID done"
