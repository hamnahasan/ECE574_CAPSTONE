#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J ablation
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --array=0-6
#SBATCH -o logs/frontier-%x-%A_%a.out
#SBATCH -e logs/frontier-%x-%A_%a.err

set -euo pipefail

source frontier/frontier_env.sh

MODALITIES=(s1 s2 dem s1_s2 s1_dem s2_dem s1_s2_dem)
MOD="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints results/logs

echo "Ablation: $MOD  Array index: $SLURM_ARRAY_TASK_ID"
echo "Attempt: ${SLURM_RESTART_COUNT:-0}  Node: ${SLURMD_NODENAME:-unknown}"
echo "Started: $(date)"

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/train_ablation.py \
        --modalities "$MOD" \
        --data_root "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir results \
        --epochs 100 \
        --batch_size 8 \
        --lr 2e-4 \
        --crop_size 256 \
        --num_workers 8 \
        --save_every 5 \
        --auto_resume

echo "Ablation $MOD finished: $(date)"

