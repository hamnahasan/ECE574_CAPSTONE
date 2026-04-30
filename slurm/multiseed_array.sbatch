#!/bin/bash
#SBATCH --job-name=multiseed
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
#SBATCH --array=0-2                         # 3 seeds by default; raise to 0-4 for n=5
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# --------------------------------------------------------------------------
# Generic multi-seed training harness.
#
# Wraps any of the project's training scripts (train_trimodal.py,
# train_fusion.py, train_bimodal.py, train_ablation.py) and runs the
# configured seed list as a SLURM array. Each array task gets one seed
# and produces independent checkpoints + logs under
#   results/checkpoints/{run_name}_seed{NNN}_*.pt
#   results/logs/{run_name}_seed{NNN}_*.json
#
# Caller controls everything via env vars:
#   TRAIN_SCRIPT      Path to the python entrypoint (required)
#   RUN_NAME          Base run name; seed is appended (required)
#   EXTRA_ARGS        Any extra args passed verbatim to the script (required;
#                     must include --modalities for ablation/bimodal scripts,
#                     --mod_dropout, etc.)
#   SEEDS             Whitespace-separated seeds; default "42 123 7"
#   EPOCHS            Default 100
#   BATCH_SIZE        Default 4
#   LR                Default 2e-4
#
# Submission examples:
#
#   # n=3 TriModal at p=0.1 (the headline run)
#   RUN_NAME=trimodal_p010 \
#   TRAIN_SCRIPT=scripts/train_trimodal.py \
#   EXTRA_ARGS="--mod_dropout 0.1" \
#     sbatch slurm/multiseed_array.sbatch
#
#   # n=3 BimodalCrossAttnUNet S1+DEM
#   RUN_NAME=bimodal_s1_dem \
#   TRAIN_SCRIPT=scripts/train_bimodal.py \
#   EXTRA_ARGS="--modalities s1_dem" \
#     sbatch slurm/multiseed_array.sbatch
#
#   # n=3 EarlyFusionUNet ablation cell
#   RUN_NAME=ablation_s1_s2_dem \
#   TRAIN_SCRIPT=scripts/train_ablation.py \
#   EXTRA_ARGS="--modalities s1_s2_dem" \
#     sbatch slurm/multiseed_array.sbatch
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Defaults (overridable via env at submit time)
SEEDS=${SEEDS:-"42 123 7"}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-2e-4}
SAVE_EVERY=${SAVE_EVERY:-5}
NUM_WORKERS=${NUM_WORKERS:-8}
CROP_SIZE=${CROP_SIZE:-256}

# Required env (fail loud if missing)
: "${TRAIN_SCRIPT:?must set TRAIN_SCRIPT, e.g. scripts/train_trimodal.py}"
: "${RUN_NAME:?must set RUN_NAME, e.g. trimodal_p010}"
: "${EXTRA_ARGS:?must set EXTRA_ARGS (script-specific flags)}"

# Pick the seed for this array task
SEED_ARRAY=($SEEDS)
if [[ $SLURM_ARRAY_TASK_ID -ge ${#SEED_ARRAY[@]} ]]; then
    echo "Array task $SLURM_ARRAY_TASK_ID >= number of seeds (${#SEED_ARRAY[@]}); exiting"
    exit 0
fi
SEED=${SEED_ARRAY[$SLURM_ARRAY_TASK_ID]}
SEED_TAG=$(printf "%03d" $SEED)
FULL_RUN_NAME="${RUN_NAME}_seed${SEED_TAG}"

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

# Allow overrides via env (e.g., for LORO splits)
DATA_ROOT=${DATA_ROOT_OVERRIDE:-$DATA_ROOT}
SPLITS_DIR=${SPLITS_DIR_OVERRIDE:-$SPLITS_DIR}

cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints results/logs

module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "=================================================="
echo "Multi-seed run: $FULL_RUN_NAME"
echo "Script:   $TRAIN_SCRIPT"
echo "Seed:     $SEED  (array index $SLURM_ARRAY_TASK_ID of ${#SEED_ARRAY[@]})"
echo "Epochs:   $EPOCHS    Batch: $BATCH_SIZE    LR: $LR"
echo "Data:     $DATA_ROOT"
echo "Splits:   $SPLITS_DIR"
echo "Extra:    $EXTRA_ARGS"
echo "Attempt:  ${SLURM_RESTART_COUNT:-0}    Started: $(date)"
echo "=================================================="

# Build the command. We use eval to expand $EXTRA_ARGS without quoting issues.
# Note: train_ablation.py uses run_name=f"ablation_{modalities}" hardcoded —
# so for that script the --run_name flag is ignored and we have to handle
# it differently. We detect that and set RUN_NAME accordingly.
if [[ "$TRAIN_SCRIPT" == *"train_ablation.py"* ]]; then
    # Ablation script doesn't accept --run_name; but it derives a unique
    # name per modality. We still want unique names per seed; the simplest
    # path is to set output_dir to a seed-tagged directory so checkpoints
    # don't collide.
    eval python "$TRAIN_SCRIPT" \
        --data_root  "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir "results/seeds/seed${SEED_TAG}" \
        --epochs     "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr         "$LR" \
        --crop_size  "$CROP_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_every "$SAVE_EVERY" \
        --seed       "$SEED" \
        --auto_resume \
        $EXTRA_ARGS
else
    eval python "$TRAIN_SCRIPT" \
        --data_root  "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir results \
        --run_name   "$FULL_RUN_NAME" \
        --epochs     "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr         "$LR" \
        --crop_size  "$CROP_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_every "$SAVE_EVERY" \
        --seed       "$SEED" \
        --auto_resume \
        $EXTRA_ARGS
fi

echo "Finished $FULL_RUN_NAME at $(date)"
