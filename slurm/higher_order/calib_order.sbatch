#!/bin/bash
#SBATCH --job-name=caliborder
#SBATCH --partition=public
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=caliborder_%j.log

# Depth-order study, step 1: calibrate the order-4 and order-5 datasets (CPU only, no training).
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export CUDA_VISIBLE_DEVICES=""
python calib_order.py
