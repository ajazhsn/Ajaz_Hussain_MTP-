#!/bin/bash
# ---------------------------------------------------------------------------------------------
# CPU-only job on WSAI-05 (40 cores): many workers, each on a few cores, restricted to experiments
# whose models are small enough to train well on a CPU. Works alongside the GPU jobs on the same
# task list (claim files prevent any task from running twice).
#
#   STAGES=tier1,tier2,tier3 ONLY=E3_rank,E4_sparsity,E7 sbatch submit_cpu.sbatch
# ---------------------------------------------------------------------------------------------
#SBATCH --job-name=lcl-cpu
#SBATCH --partition=public
#SBATCH --nodelist=WSAI-05
#SBATCH --cpus-per-task=36
#SBATCH --mem=100G
#SBATCH --time=23:50:00
#SBATCH --output=logs/lcl_cpu_%j.out

source /storage/opt/miniforge3/etc/profile.d/conda.sh
conda activate "${LCL_ENV:-pytorch-env}"

cd "$SLURM_SUBMIT_DIR"
STAGES=${STAGES:-pilot}
WORKERS=${CPU_WORKERS:-9}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-36} / WORKERS ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export CUDA_VISIBLE_DEVICES=""

echo "host $(hostname)  stages $STAGES  only=${ONLY:-all}  workers $WORKERS  threads/worker $OMP_NUM_THREADS"
for w in $(seq 1 "$WORKERS"); do
    python scripts/run.py --stage "$STAGES" ${RETRY:+--retry-errors} ${ONLY:+--only $ONLY} &
    sleep 5
done
wait
echo "all workers finished"
