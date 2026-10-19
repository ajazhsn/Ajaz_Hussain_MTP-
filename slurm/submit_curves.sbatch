#!/bin/bash
# Training curves: every model once on the reference data, loss recorded after every epoch.
#   sbatch --array=0-1 slurm/submit_curves.sbatch        (2 GPUs x 4 workers; about 1 hour)
#   python scripts/curves.py --status                     (progress)
#   python scripts/curves.py --merge                      (when all are done: writes results/curves.json)
#SBATCH --job-name=lcl-curves
#SBATCH --partition=public
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lcl_curves_%A_%a.out
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
for w in 1 2 3 4; do python scripts/curves.py & sleep 5; done
wait
echo "all workers finished"
