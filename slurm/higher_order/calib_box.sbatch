#!/bin/bash
#SBATCH --job-name=calibbox
#SBATCH --partition=public
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=calibbox_%j.log

# Step 2a: calibrate the box-kernel datasets (CPU only, no training, ~20-40 min).
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
export CUDA_VISIBLE_DEVICES=""
python calib_box.py
