#!/bin/bash
#SBATCH --job-name=trimodal_weak
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:v100s:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Paper extension: weakly-labeled pretraining for TriModal Fusion U-Net
# (S1 + S2 + DEM cross-attention).
#
# Phase A: pretrain on ~4 385 WeaklyLabeled chips (30 epochs, lr=2e-4)
# Phase B: fine-tune on 446 HandLabeled chips (60 epochs, lr=4e-5)
# Eval:    test split of HandLabeled
#
# Prerequisites:
#   - DEM aligned to all WeaklyLabeled chips under DEMWeak/ on Isaac
#     (the upstream alignment job must have produced Bolivia/Ghana/.../*_DEMWeak.tif)
#   - Weak split CSVs at $WEAK_SPLITS/flood_train_data.csv and flood_valid_data.csv
#
# Notes:
#   - WEAK_LABEL_SUBDIR defaults to S1OtsuLabelWeak (cloud-immune labels).
#     Override to S2IndexLabelWeak if you want cloud-sensitive spectral labels.
#   - pretrain_batch_size = 4 to fit V100s 32GB memory budget for 3-modal
#     model with cross-attention at 4 scales.
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# -- HandLabeled paths (used for fine-tune + final test) -------------------
HAND_DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
HAND_SPLITS=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled

# -- WeaklyLabeled paths (used for pretrain) -------------------------------
WEAK_DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/WeaklyLabeled
WEAK_SPLITS=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_weakly_labeled

WEAK_TRAIN_CSV=flood_train_data.csv
WEAK_VAL_CSV=flood_valid_data.csv

# Sen1Floods11 v1.1 ships the weakly-labeled labels under TWO directories:
#   S1OtsuLabelWeak    -- Otsu threshold on S1 VH (cloud-immune; default)
#   S2IndexLabelWeak   -- spectral index threshold on S2 (cloud-sensitive)
WEAK_LABEL_SUBDIR=S1OtsuLabelWeak

PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

# Per-seed overrides for n=3 parity with other TriModal runs.
# Default seed=42 / run_name=trimodal_weakpretrain; override at submit:
#   SEED=123 RUN_NAME=trimodal_weakpretrain_seed123 sbatch ...
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-trimodal_weakpretrain}

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
    "$HAND_DATA_ROOT/DEMHand" \
    "$HAND_DATA_ROOT/LabelHand" \
    "$HAND_SPLITS/flood_train_data.csv" \
    "$HAND_SPLITS/flood_valid_data.csv" \
    "$HAND_SPLITS/flood_test_data.csv" \
    "$WEAK_DATA_ROOT/S1Weak" \
    "$WEAK_DATA_ROOT/S2Weak" \
    "$WEAK_DATA_ROOT/DEMWeak" \
    "$WEAK_DATA_ROOT/$WEAK_LABEL_SUBDIR" \
    "$WEAK_SPLITS/$WEAK_TRAIN_CSV" \
    "$WEAK_SPLITS/$WEAK_VAL_CSV"
do
    if [[ ! -e "$path" ]]; then
        echo "MISSING: $path"; fail=1
    fi
done
if [[ $fail -ne 0 ]]; then
    echo ""
    echo "One or more required paths is missing. Most likely DEMWeak alignment"
    echo "hasn't completed for the WeaklyLabeled chip grid. Check the upstream"
    echo "align_dem_weak job and re-submit once DEMWeak/ is populated."
    exit 1
fi

python scripts/pretrain_finetune_trimodal.py \
    --hand_data_root    "$HAND_DATA_ROOT" \
    --hand_splits_dir   "$HAND_SPLITS" \
    --weak_data_root    "$WEAK_DATA_ROOT" \
    --weak_splits_dir   "$WEAK_SPLITS" \
    --weak_train_csv    "$WEAK_TRAIN_CSV" \
    --weak_val_csv      "$WEAK_VAL_CSV" \
    --weak_label_subdir "$WEAK_LABEL_SUBDIR" \
    --output_dir        results \
    --run_name          "$RUN_NAME" \
    --pretrain_epochs  30 \
    --pretrain_lr      2e-4 \
    --pretrain_batch_size 4 \
    --finetune_epochs  60 \
    --finetune_lr      4e-5 \
    --finetune_batch_size 4 \
    --crop_size        256 \
    --num_workers      8 \
    --save_every       5 \
    --seed             "$SEED" \
    --auto_resume

echo "Finished at $(date)"
