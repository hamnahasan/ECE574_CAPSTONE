#!/bin/bash
#SBATCH --job-name=robustness
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Modality-robustness evaluation: for each TriModal checkpoint produced by
# slurm/sensitivity_mod_dropout.sbatch, run the test split with each
# modality individually zeroed at inference. Produces 6 JSON files, one
# per `p`, each with 7 scenarios.
#
# Run AFTER all six sensitivity-sweep training jobs have completed.
# Wall-time: ~7 scenarios x 90 chips x 6 runs x ~2s/chip ~ 30 min.
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

DATA_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled
SPLITS_DIR=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd $PROJECT_DIR
mkdir -p logs results/logs

module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "Started at $(date)"

for TAG in 000 005 010 020 030 050; do
    RUN_NAME="trimodal_p${TAG}"
    CKPT="results/checkpoints/${RUN_NAME}_best.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "SKIP: $RUN_NAME (checkpoint missing: $CKPT)"
        continue
    fi
    echo ""
    echo "=== Evaluating $RUN_NAME ==="
    python scripts/eval_modality_robustness.py \
        --run_name   "$RUN_NAME" \
        --data_root  "$DATA_ROOT" \
        --splits_dir "$SPLITS_DIR" \
        --output_dir results \
        --split      test \
        --num_workers 4
done

echo ""
echo "Finished at $(date)"
