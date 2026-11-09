#!/bin/bash
#SBATCH --job-name=interpA
#SBATCH --partition=public
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=interp_analysis_%j.log

# Interpretation, step 2: run the four analyses on the saved weights.
#   Output: interp_report.txt (tables) and interp_out/interp_results.json (all rows)
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
python interpret.py > interp_report.txt 2>&1
echo "done - see interp_report.txt"
