#!/bin/bash
#SBATCH --job-name=sens_moddrop
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
#SBATCH --array=0-5                    # 6 values of p, fits Isaac QOS cap
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

# --------------------------------------------------------------------------
# Sensitivity analysis on modality-dropout probability p.
#
# Trains TriModal U-Net (S1+S2+DEM, 3-way cross-attention) at six values of
# the modality-dropout rate, holding everything else identical (seed, epochs,
# lr, batch size, splits). Each run produces its own checkpoint and history
# under results/checkpoints/trimodal_p{NNN}_*.pt and
# results/logs/trimodal_p{NNN}_history.json.
#
# Eval (in-distribution AND with one modality zeroed at inference) is run
# separately by scripts/eval_modality_robustness.py once all six finish.
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Sweep grid — kept small enough to fit Isaac's 6-job QOS limit.
P_VALUES=(0.0 0.05 0.10 0.20 0.30 0.50)
P_TAGS=(000 005 010 020 030 050)

P_VAL=${P_VALUES[$SLURM_ARRAY_TASK_ID]}
P_TAG=${P_TAGS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="trimodal_p${P_TAG}"

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
mkdir -p logs results/checkpoints results/logs

module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "Sensitivity sweep: p=$P_VAL  run=$RUN_NAME  (array index $SLURM_ARRAY_TASK_ID)"
echo "Attempt: ${SLURM_RESTART_COUNT:-0}  Started: $(date)"

python scripts/train_trimodal.py \
    --data_root    "$DATA_ROOT" \
    --splits_dir   "$SPLITS_DIR" \
    --output_dir   results \
    --run_name     "$RUN_NAME" \
    --epochs       100 \
    --batch_size   4 \
    --lr           2e-4 \
    --crop_size    256 \
    --num_workers  8 \
    --save_every   5 \
    --mod_dropout  "$P_VAL" \
    --seed         42 \
    --auto_resume

echo "Run $RUN_NAME (p=$P_VAL) finished at $(date)"
