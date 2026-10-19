#!/bin/bash
# ---------------------------------------------------------------------------------------------
# WSAI cluster GPU job: each array element = 1 V100; each GPU runs WORKERS_PER_GPU copies of run.py.
# Copies coordinate through claim files, so you can submit this as often as you like.
#
# Usage (from the repo root, e.g. /storage_server/da25m006/Ajaz_Hussain_MTP-):
#   mkdir -p logs
#   STAGES=pilot,tune sbatch --array=0-3 slurm/submit.sbatch
#   STAGES=tier1,tier2,tier3 sbatch --array=0-3 slurm/submit.sbatch
# Your limit is 4 GPUs / 4 running jobs / 24 h per job. Pending jobs simply wait for a free V100.
# (The K80 nodes currently have a broken driver, so this asks for V100s only.)
# ---------------------------------------------------------------------------------------------
#SBATCH --job-name=lcl
#SBATCH --partition=public
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --output=logs/lcl_%A_%a.out

source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate "${LCL_ENV:-pytorch-env}"

cd "$SLURM_SUBMIT_DIR"
STAGES=${STAGES:-pilot}
WPG=${WORKERS_PER_GPU:-4}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / WPG ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "host $(hostname)  stages $STAGES  workers/GPU $WPG  threads/worker $OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for w in $(seq 1 "$WPG"); do
    python scripts/run.py --stage "$STAGES" ${RETRY:+--retry-errors} ${ONLY:+--only $ONLY} &
    sleep 5
done
wait
echo "all workers finished"
