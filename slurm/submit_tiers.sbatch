#!/bin/bash
#SBATCH --job-name=lcl-tiers
#SBATCH --partition=public
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --output=logs/lcl_tiers_%A_%a.out
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR"
# the first job to arrive picks the tuned settings; the others wait until the file is complete
if mkdir results/.finalize_lock 2>/dev/null; then
  python scripts/run.py --finalize-tuning > logs/finalize_tuning.txt 2>&1
fi
for i in $(seq 1 120); do
  python -c "import json; json.load(open('results/hp.json'))" 2>/dev/null && break
  sleep 10
done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
for w in 1 2 3 4; do python scripts/run.py --stage tier1,tier2,tier3 & sleep 5; done
wait
