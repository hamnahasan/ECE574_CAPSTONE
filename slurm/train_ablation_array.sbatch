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
#SBATCH --array=0-5                    # 6 of 7 variants (Isaac QOS caps at 6
                                       # SUBMITTED jobs total). The 7th variant
                                       # (dem alone) is submitted separately
                                       # via slurm/train_ablation_single.sbatch
                                       # once a slot frees up. Variants are
                                       # ordered below from most important to
                                       # least important, so the deferred one
                                       # is the weakest ablation.
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# --------------------------------------------------------------------------
# Ablation study — 7 modality subsets. Isaac campus-gpu QOS caps at 6
# submitted jobs total, so we submit 6 here and defer the 7th (dem alone)
# to train_ablation_single.sbatch — run after any of these finishes.
# All variants auto-resume independently across the 24hr wall-time.
#
# Variants — ordered so the LEAST critical (dem alone) is at index 6
# and therefore excluded from the array. Indices 0-5 run:
#   0: s1_s2_dem — Phase 3 equivalent via early fusion (most critical)
#   1: s1_s2     — Phase 2 equivalent via early fusion
#   2: s1        — SAR only (architecture-controlled FCN comparison)
#   3: s1_dem    — tests DEM value without optical
#   4: s2_dem    — tests if SAR is necessary
#   5: s2        — optical only
# Deferred to train_ablation_single.sbatch:
#   6: dem       — topography only (expected worst; sanity check)
# --------------------------------------------------------------------------

set -euo pipefail

# Silence Isaac's OpenMP warning about --cpus-per-task. We do not use OpenMP
# explicitly, but numpy/scikit-image may use it internally. Setting this
# explicitly acknowledges the warning and ensures BLAS doesn't oversubscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Modality list — reordered so dem (weakest, deferred) is last.
# train_ablation_single.sbatch uses the same list and picks index 6.
MODALITIES=(s1_s2_dem s1_s2 s1 s1_dem s2_dem s2 dem)
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
