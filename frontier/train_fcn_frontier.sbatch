#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J fcn_s1
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/frontier-%x-%j.out
#SBATCH -e logs/frontier-%x-%j.err

set -euo pipefail

source frontier/frontier_env.sh

cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints results/logs

echo "Job: $SLURM_JOB_NAME  Attempt: ${SLURM_RESTART_COUNT:-0}  Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/train_fcn_baseline.py \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir results \
        --run_name fcn_baseline \
        --epochs 100 \
        --batch_size 8 \
        --lr 2e-4 \
        --crop_size 256 \
        --num_workers 8 \
        --save_every 5 \
        --auto_resume

echo "Finished: $(date)"

