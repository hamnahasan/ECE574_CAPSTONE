#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J eval_bolivia
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o logs/frontier-%x-%j.out
#SBATCH -e logs/frontier-%x-%j.err

set -euo pipefail

source frontier/frontier_env.sh

cd "$PROJECT_DIR"
mkdir -p logs results/logs

echo "Started Bolivia evaluation: $(date)"

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/evaluate.py \
        --model fusion \
        --checkpoint results/checkpoints/fusion_unet_best.pt \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --split bolivia \
        --output results/logs/fusion_unet_bolivia_results.json

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/evaluate.py \
        --model trimodal \
        --checkpoint results/checkpoints/trimodal_unet_best.pt \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --split bolivia \
        --output results/logs/trimodal_unet_bolivia_results.json

echo "Finished Bolivia evaluation: $(date)"

