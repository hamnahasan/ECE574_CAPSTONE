#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J core_packed
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/frontier-%x-%j.out
#SBATCH -e logs/frontier-%x-%j.err

set -euo pipefail

source frontier/frontier_env.sh

cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints results/logs

echo "Packed core training started: $(date)"
echo "Node: ${SLURMD_NODENAME:-unknown}  Attempt: ${SLURM_RESTART_COUNT:-0}"

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

run_one() {
    local name="$1"
    shift
    (
        export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}-${name}"
        export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
        mkdir -p "$MIOPEN_USER_DB_PATH"
        echo "[$name] started: $(date)"
        srun --exclusive -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest "$@" \
            > "logs/frontier-${name}-${SLURM_JOB_ID}.out" \
            2> "logs/frontier-${name}-${SLURM_JOB_ID}.err"
        echo "[$name] finished: $(date)"
    ) &
}

run_one fcn \
    python -u scripts/train_fcn_baseline.py \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir results \
        --run_name fcn_baseline \
        --epochs 100 \
        --batch_size 8 \
        --lr 2e-4 \
        --crop_size 256 \
        --num_workers 8 \
        --save_every 5 \
        --auto_resume

run_one fusion \
    python -u scripts/train_fusion.py \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir results \
        --run_name fusion_unet \
        --epochs 100 \
        --batch_size 8 \
        --lr 2e-4 \
        --crop_size 256 \
        --num_workers 8 \
        --save_every 5 \
        --auto_resume

run_one trimodal \
    python -u scripts/train_trimodal.py \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir results \
        --run_name trimodal_unet \
        --epochs 100 \
        --batch_size 4 \
        --lr 2e-4 \
        --crop_size 256 \
        --num_workers 8 \
        --save_every 5 \
        --mod_dropout 0.1 \
        --auto_resume

wait
echo "Packed core training finished: $(date)"

