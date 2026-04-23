#!/bin/bash
#SBATCH --job-name=eval_bolivia
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00                # eval is fast — 1hr is plenty
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Cross-region generalization test.
# Bolivia is held out in Sen1Floods11 — it is NOT in train/val/test.
# Evaluating the best Phase 3 model on Bolivia tests whether the learned
# flood features generalize to a geographic region never seen during training.
# A large drop from test-IoU to bolivia-IoU indicates overfitting to the
# other 10 flood events. A small drop suggests the model captures
# transferable flood physics — the ideal outcome for a disaster response tool.
# --------------------------------------------------------------------------

set -euo pipefail

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
module load cuda/12.1.1-binary

# Initialize conda for batch mode (critical for SLURM scripts)
eval "$(conda shell.bash hook)"
conda activate floodseg

# Evaluate all three main models on Bolivia
for MODEL in fusion trimodal; do
    CKPT="results/checkpoints/${MODEL}_unet_best.pt"
    if [ -f "$CKPT" ]; then
        echo "== Evaluating $MODEL on Bolivia =="
        python scripts/evaluate.py \
            --model       "$MODEL" \
            --checkpoint  "$CKPT" \
            --data_root   "$DATA_ROOT" \
            --splits_dir  "$SPLITS_DIR" \
            --split       bolivia \
            --output      "results/logs/${MODEL}_bolivia_results.json"
    else
        echo "WARNING: checkpoint not found: $CKPT"
    fi
done

# FCN baseline on Bolivia (fair comparison)
if [ -f "results/checkpoints/fcn_baseline_best.pt" ]; then
    python scripts/evaluate.py \
        --model fcn \
        --checkpoint  results/checkpoints/fcn_baseline_best.pt \
        --data_root   "$DATA_ROOT" \
        --splits_dir  "$SPLITS_DIR" \
        --split       bolivia \
        --output      results/logs/fcn_bolivia_results.json
fi

echo "Done at $(date)"
