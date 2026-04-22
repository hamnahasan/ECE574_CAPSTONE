#!/bin/bash
#SBATCH --job-name=trimodal_s1s2dem
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G                      # slightly more — 3 encoders
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Phase 3 — Tri-modal S1+S2+DEM with 3-way cross-attention + modality dropout
# Modality dropout (--mod_dropout 0.1) randomly zeros a modality during training,
# forcing the model to remain robust when a modality is missing at inference
# (e.g., cloud-obscured S2, missing DEM tiles).
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

python scripts/train_trimodal.py \
    --data_root    "$DATA_ROOT" \
    --splits_dir   "$SPLITS_DIR" \
    --output_dir   results \
    --run_name     trimodal_unet \
    --epochs       100 \
    --batch_size   4 \
    --lr           2e-4 \
    --crop_size    256 \
    --num_workers  8 \
    --save_every   5 \
    --mod_dropout  0.1 \
    --auto_resume

echo "Finished at $(date)"
