#!/bin/bash
#SBATCH --job-name=temp_scale
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100s:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Post-hoc temperature scaling (paper experiment F4): fit a single scalar
# temperature T on the val split using NLL minimization, apply at test
# time, report pre/post ECE. Reviewers expect this calibration baseline
# alongside MC Dropout.
# --------------------------------------------------------------------------

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

CHECKPOINT=${CHECKPOINT:-results/checkpoints/trimodal_p010_seed042_best.pt}

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd "$PROJECT_DIR"
mkdir -p logs results/logs

module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "Temperature scaling on $CHECKPOINT  ($(date))"

python scripts/temperature_scaling.py \
    --model trimodal \
    --checkpoint "$CHECKPOINT" \
    --data_root  "$DATA_ROOT" \
    --splits_dir "$SPLITS_DIR"

echo "Finished at $(date)"
