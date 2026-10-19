#!/bin/bash
#SBATCH --job-name=lcl-tiers-cpu
#SBATCH --partition=public
#SBATCH --nodelist=WSAI-05
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G
#SBATCH --time=23:50:00
#SBATCH --output=logs/lcl_tiers_cpu_%j.out
source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate pytorch-env
cd "$SLURM_SUBMIT_DIR"
if mkdir results/.finalize_lock 2>/dev/null; then
  python scripts/run.py --finalize-tuning > logs/finalize_tuning.txt 2>&1
fi
for i in $(seq 1 120); do
  python -c "import json; json.load(open('results/hp.json'))" 2>/dev/null && break
  sleep 10
done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
for w in $(seq 1 9); do python scripts/run.py --stage tier1,tier2,tier3 --only E3_rank,E4_sparsity,E7 & sleep 5; done
wait
