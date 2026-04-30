#!/bin/bash
#SBATCH --job-name=weak_pretrain
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Paper extension: weakly-labeled pretraining for Fusion U-Net (S1+S2 cross
# attention).
#
# Phase A: pretrain on ~4 385 WeaklyLabeled chips (30 epochs, lr=2e-4)
# Phase B: fine-tune on 446 HandLabeled chips (60 epochs, lr=4e-5)
# Eval:    test split of HandLabeled
#
# We use Fusion U-Net (NOT TriModal) here because the DEM tiles in this
# repo are aligned only to HandLabeled chips. TriModal weak pretraining
# requires an upstream DEM-alignment pass on ~4 385 weak chips first.
# That preprocessing job is separate; if/when it lands, copy this script
# and swap `train_fusion`->`train_trimodal` paths.
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# -- HandLabeled paths (used for fine-tune + final test) -------------------
HAND_DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
HAND_SPLITS=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled

# -- WeaklyLabeled paths (used for pretrain) -------------------------------
# NOTE: Sen1Floods11 v1.1 ships weak data under
#   data/flood_events/WeaklyLabeled/{S1Weak,S2Weak,LabelWeak}
# and weak split CSVs under
#   splits/flood_weakly_labeled/  (or similar)
# Confirm directory naming on Isaac before submitting — the verification
# block below will exit early with a clear error if any path is missing.
WEAK_DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/WeaklyLabeled
WEAK_SPLITS=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_weakly_labeled

# Override these if your weak split CSVs use different names.
WEAK_TRAIN_CSV=flood_train_data.csv
WEAK_VAL_CSV=flood_valid_data.csv

PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE
RUN_NAME=fusion_weakpretrain

cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints results/logs

module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "Job $SLURM_JOB_NAME [attempt ${SLURM_RESTART_COUNT:-0}] on $SLURMD_NODENAME at $(date)"

# -- Path verification: fail loud if anything is missing -------------------
fail=0
for path in \
    "$HAND_DATA_ROOT/S1Hand" \
    "$HAND_DATA_ROOT/S2Hand" \
    "$HAND_DATA_ROOT/LabelHand" \
    "$HAND_SPLITS/flood_train_data.csv" \
    "$HAND_SPLITS/flood_valid_data.csv" \
    "$HAND_SPLITS/flood_test_data.csv" \
    "$WEAK_DATA_ROOT/S1Weak" \
    "$WEAK_DATA_ROOT/S2Weak" \
    "$WEAK_DATA_ROOT/LabelWeak" \
    "$WEAK_SPLITS/$WEAK_TRAIN_CSV" \
    "$WEAK_SPLITS/$WEAK_VAL_CSV"
do
    if [[ ! -e "$path" ]]; then
        echo "MISSING: $path"; fail=1
    fi
done
if [[ $fail -ne 0 ]]; then
    echo ""
    echo "One or more required paths is missing. Update WEAK_DATA_ROOT /"
    echo "WEAK_SPLITS / WEAK_*_CSV in this script to match the Isaac layout,"
    echo "then re-submit. (Sen1Floods11 v1.1 sometimes uses 'flood_weak_data.csv'"
    echo "or names like '_perm_water'.)"
    exit 1
fi

python scripts/pretrain_finetune_fusion.py \
    --hand_data_root   "$HAND_DATA_ROOT" \
    --hand_splits_dir  "$HAND_SPLITS" \
    --weak_data_root   "$WEAK_DATA_ROOT" \
    --weak_splits_dir  "$WEAK_SPLITS" \
    --weak_train_csv   "$WEAK_TRAIN_CSV" \
    --weak_val_csv     "$WEAK_VAL_CSV" \
    --output_dir       results \
    --run_name         "$RUN_NAME" \
    --pretrain_epochs  30 \
    --pretrain_lr      2e-4 \
    --pretrain_batch_size 8 \
    --finetune_epochs  60 \
    --finetune_lr      4e-5 \
    --finetune_batch_size 4 \
    --crop_size        256 \
    --num_workers      8 \
    --save_every       5 \
    --seed             42 \
    --auto_resume

echo "Finished at $(date)"
