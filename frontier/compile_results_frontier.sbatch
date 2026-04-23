#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J compile_results
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -o logs/frontier-%x-%j.out
#SBATCH -e logs/frontier-%x-%j.err

set -euo pipefail

source frontier/frontier_env.sh

cd "$PROJECT_DIR"
mkdir -p logs results/logs

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

python -u scripts/compile_results.py \
    --logs_dir results/logs \
    --output_csv results/logs/all_results.csv \
    --output_md results/logs/all_results.md

