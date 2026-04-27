#!/bin/bash
#SBATCH --job-name=mc_unc_bolivia
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00              # Bolivia is only 15 chips — 30 min is plenty
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# MC Dropout uncertainty evaluation on the BOLIVIA cross-region holdout.
# Identical to mc_uncertainty.sbatch except --split bolivia. Reports ECE +
# reliability diagram for cross-region calibration. Lets us answer "is the
# model still well-calibrated when it's out of distribution?" with a number
# instead of a hand-wave.
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

python scripts/mc_uncertainty.py \
    --checkpoint  results/checkpoints/trimodal_unet_best.pt \
    --model       trimodal \
    --data_root   "$DATA_ROOT" \
    --splits_dir  "$SPLITS_DIR" \
    --split       bolivia \
    --n_samples   20 \
    --output_dir  results

echo "Bolivia MC uncertainty done at $(date)"
