#!/bin/bash
#SBATCH --job-name=ablation_dem
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Leftover ablation variant: DEM only (index 6).
#
# This variant is deferred because Isaac's campus-gpu QOS limits users to
# 6 submitted jobs at once, and the main array already uses all 6 slots.
#
# SUBMIT THIS ONCE: a slot frees from the array (any of the 6 array tasks
# either finishes, fails, or is cancelled). Check free slots with:
#     squeue -u $USER | wc -l      # should be < 6 before submitting
#
# Or use slurm/submit_ablation_all.sh which polls and submits automatically.
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MOD=dem

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
mkdir -p logs results/checkpoints results/logs

module load cuda/12.1.1-binary

# Initialize conda for batch mode
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "Ablation $MOD [attempt ${SLURM_RESTART_COUNT:-0}] on $SLURMD_NODENAME at $(date)"

python scripts/train_ablation.py \
    --modalities   "$MOD" \
    --data_root    "$DATA_ROOT" \
    --splits_dir   "$SPLITS_DIR" \
    --output_dir   results \
    --epochs       100 \
    --batch_size   8 \
    --lr           2e-4 \
    --crop_size    256 \
    --num_workers  8 \
    --save_every   5 \
    --auto_resume

echo "Ablation $MOD finished at $(date)"
