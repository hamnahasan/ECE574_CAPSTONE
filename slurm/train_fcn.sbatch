#!/bin/bash
#SBATCH --job-name=fcn_s1
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Phase 1 baseline — FCN-ResNet50 on S1 (VV/VH) only.
# Replicates the deep-learning baseline from Bonafilia et al. (2020).
# Trained on Isaac so all results in the paper come from the same environment
# and the compile_results.py aggregator picks up every cell uniformly.
# --------------------------------------------------------------------------

set -euo pipefail

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
mkdir -p logs results/checkpoints results/logs

module load anaconda3/2024.02 cuda/12.1
source activate floodseg

echo "Job $SLURM_JOB_NAME [attempt ${SLURM_RESTART_COUNT:-0}] on $SLURMD_NODENAME at $(date)"

python scripts/train_fcn_baseline.py \
    --data_root   "$DATA_ROOT" \
    --splits_dir  "$SPLITS_DIR" \
    --output_dir  results \
    --run_name    fcn_baseline \
    --epochs      100 \
    --batch_size  8 \
    --lr          2e-4 \
    --crop_size   256 \
    --num_workers 8 \
    --save_every  5 \
    --auto_resume

echo "Finished at $(date)"
