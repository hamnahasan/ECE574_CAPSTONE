#!/bin/bash
#SBATCH --job-name=fusion_s1s2
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue                      # allow SLURM to auto-requeue on timeout
#SBATCH --signal=B:USR1@300            # warn us 5min before wall-time
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Phase 2 — Dual-encoder S1+S2 cross-attention fusion
# Auto-resumes from results/checkpoints/fusion_unet_latest.pt if present.
# When the 24hr wall-time is hit, SLURM requeues the same job and training
# picks up seamlessly from the latest checkpoint (epoch, optimizer, LR, RNG).
# --------------------------------------------------------------------------

set -euo pipefail

# UPDATE THESE PATHS for your Isaac account if different
DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
mkdir -p logs results/checkpoints results/logs

# Environment (see scripts/setup_isaac.sh)
module load anaconda3/2024.02 cuda/12.1
source activate floodseg

echo "===================================================="
echo "Job:       $SLURM_JOB_NAME"
echo "JobID:     $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:   $(date)"
echo "Attempt:   ${SLURM_RESTART_COUNT:-0}"
echo "===================================================="

python scripts/train_fusion.py \
    --data_root   "$DATA_ROOT" \
    --splits_dir  "$SPLITS_DIR" \
    --output_dir  results \
    --run_name    fusion_unet \
    --epochs      100 \
    --batch_size  8 \
    --lr          2e-4 \
    --crop_size   256 \
    --num_workers 8 \
    --save_every  5 \
    --auto_resume

echo "===================================================="
echo "Finished:  $(date)"
echo "===================================================="
