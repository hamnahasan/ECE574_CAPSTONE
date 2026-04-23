#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J mc_uncertainty
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o logs/frontier-%x-%j.out
#SBATCH -e logs/frontier-%x-%j.err

set -euo pipefail

source frontier/frontier_env.sh

cd "$PROJECT_DIR"
mkdir -p logs results/logs results/figures

echo "Started MC uncertainty: $(date)"

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/mc_uncertainty.py \
        --checkpoint results/checkpoints/trimodal_unet_best.pt \
        --model trimodal \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --split test \
        --n_samples 20 \
        --n_bins 15 \
        --output_dir results \
        --n_save_maps 6

echo "Finished MC uncertainty: $(date)"

