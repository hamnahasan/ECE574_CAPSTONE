#!/bin/bash
#SBATCH --job-name=mc_uncertainty
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# MC Dropout uncertainty evaluation on the test set.
# Runs N=20 forward passes per chip with dropout kept active. Reports ECE
# (Expected Calibration Error) and saves a reliability diagram. Low ECE
# means the model's confidence scores accurately reflect its correctness —
# a critical property for disaster response decision support.
# --------------------------------------------------------------------------

set -euo pipefail

# Silence OpenMP warning from Isaac for --cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
module load cuda/12.1.1-binary

# Initialize conda for batch mode (critical for SLURM scripts)
eval "$(conda shell.bash hook)"
conda activate floodseg

python scripts/mc_uncertainty.py \
    --checkpoint  results/checkpoints/trimodal_unet_best.pt \
    --model       trimodal \
    --data_root   "$DATA_ROOT" \
    --splits_dir  "$SPLITS_DIR" \
    --split       test \
    --n_samples   20 \
    --output_dir  results

echo "MC uncertainty done at $(date)"
