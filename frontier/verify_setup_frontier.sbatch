#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J verify_floodseg
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o logs/frontier-%x-%j.out
#SBATCH -e logs/frontier-%x-%j.err

set -euo pipefail

source frontier/frontier_env.sh

cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints results/logs

echo "Job:     $SLURM_JOB_NAME"
echo "JobID:   $SLURM_JOB_ID"
echo "Node:    ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/verify_setup.py \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR"

echo "Finished: $(date)"

