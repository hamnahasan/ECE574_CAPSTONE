#!/bin/bash
#SBATCH --job-name=mc_n_sweep
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100s:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# MC-Dropout N sweep (paper experiment F2): validate the N=20 default by
# measuring ECE / NLL / Brier at N in {5, 10, 20, 50, 100} on the headline
# TriModal checkpoint. Produces a single-figure plot of ECE vs N which the
# paper uses to defend the N=20 choice.
# --------------------------------------------------------------------------

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

CHECKPOINT=${CHECKPOINT:-results/checkpoints/trimodal_p010_seed042_best.pt}
SPLIT=${SPLIT:-test}

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd "$PROJECT_DIR"
mkdir -p logs results/logs results/figures

module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "MC-N sweep on $CHECKPOINT  ($(date))"

python scripts/mc_n_sweep.py \
    --model trimodal \
    --checkpoint "$CHECKPOINT" \
    --data_root  "$DATA_ROOT" \
    --splits_dir "$SPLITS_DIR" \
    --split      "$SPLIT" \
    --n_values   5 10 20 50 100

echo "Finished at $(date)"
