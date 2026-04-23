#!/bin/bash
#SBATCH -A PROJECT_ID_HERE
#SBATCH -J ablation_packed
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/frontier-%x-%j.out
#SBATCH -e logs/frontier-%x-%j.err

set -euo pipefail

source frontier/frontier_env.sh

cd "$PROJECT_DIR"
mkdir -p logs results/checkpoints results/logs

echo "Packed ablation training started: $(date)"
echo "Node: ${SLURMD_NODENAME:-unknown}  Attempt: ${SLURM_RESTART_COUNT:-0}"

# Activate conda environment via direct path (critical for SLURM batch mode)
eval "$(${FRONTIER_ENV}/bin/conda shell.bash hook)"
conda activate "$FRONTIER_ENV"

run_ablation() {
    local mod="$1"
    (
        export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}-${mod}"
        export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
        mkdir -p "$MIOPEN_USER_DB_PATH"
        echo "[$mod] started: $(date)"
        srun --exclusive -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
            python -u scripts/train_ablation.py \
                --modalities "$mod" \
                --data_root "$DATA_ROOT" \
                --splits_dir "$SPLITS_DIR" \
                --output_dir results \
                --epochs 100 \
                --batch_size 8 \
                --lr 2e-4 \
                --crop_size 256 \
                --num_workers 8 \
                --save_every 5 \
                --auto_resume \
            > "logs/frontier-ablation-${mod}-${SLURM_JOB_ID}.out" \
            2> "logs/frontier-ablation-${mod}-${SLURM_JOB_ID}.err"
        echo "[$mod] finished: $(date)"
    ) &
}

for mod in s1 s2 dem s1_s2 s1_dem s2_dem s1_s2_dem; do
    run_ablation "$mod"
done

wait
echo "Packed ablation training finished: $(date)"

