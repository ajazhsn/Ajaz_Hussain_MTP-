#!/bin/bash
#SBATCH --job-name=interp2A
#SBATCH --partition=public
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=interp2_analysis_%j.log

# Analyses on all 5 datasets x 10 seeds. Output: interp_report_10seeds.txt, interp_out10/interp_results.json
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR/src/higher_order"      # submit from the repo root: sbatch slurm/higher_order/<script>
python interpret.py --datasets D0_lcl,D2_comp,D3_centre,D2b_comp,D3b_centre --seeds 0-9 \
    --out interp_out10 > interp_report_10seeds.txt 2>&1
echo "done - see interp_report_10seeds.txt"
