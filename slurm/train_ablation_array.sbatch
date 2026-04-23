#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-6%6                  # 7 ablation variants; %6 caps concurrent at 6
                                       # (Isaac max 6 jobs/user; 7th queues and
                                       #  auto-starts when a slot opens)
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# --------------------------------------------------------------------------
# Ablation study — 7 modality subsets run as a SLURM job array.
# Each $SLURM_ARRAY_TASK_ID maps to one modality combination. The array
# syntax lets Isaac schedule all 7 on different GPUs in parallel (subject
# to queue limits). Each run auto-resumes independently.
#
# Modality combinations tested:
#   0: s1        — SAR only (reproduce FCN baseline)
#   1: s2        — optical only
#   2: dem       — topography only (expected poor)
#   3: s1_s2     — Phase 2 equivalent via early fusion
#   4: s1_dem    — tests DEM value without optical
#   5: s2_dem    — tests if SAR is necessary
#   6: s1_s2_dem — Phase 3 equivalent via early fusion
# --------------------------------------------------------------------------

set -euo pipefail

# Modality list matches the indices above.
MODALITIES=(s1 s2 dem s1_s2 s1_dem s2_dem s1_s2_dem)
MOD=${MODALITIES[$SLURM_ARRAY_TASK_ID]}

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
mkdir -p logs results/checkpoints results/logs

module load cuda/12.1.1-binary

# Initialize conda for batch mode (critical for SLURM scripts)
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "Ablation variant: $MOD  (array index $SLURM_ARRAY_TASK_ID)"
echo "Attempt: ${SLURM_RESTART_COUNT:-0}  Started: $(date)"

python scripts/train_ablation.py \
    --modalities   "$MOD" \
    --data_root    "$DATA_ROOT" \
    --splits_dir   "$SPLITS_DIR" \
    --output_dir   results \
    --epochs       100 \
    --batch_size   8 \
    --lr           2e-4 \
    --crop_size    256 \
    --num_workers  8 \
    --save_every   5 \
    --auto_resume

echo "Ablation $MOD finished at $(date)"
