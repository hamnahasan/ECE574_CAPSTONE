#!/bin/bash
#SBATCH --job-name=render_figs
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00            # rendering 3 figures is fast (~10 min)
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Render the 3 presentation figures on a GPU node:
#   1. ablation_s1_s2 best/worst test predictions
#   2. trimodal best/worst Bolivia predictions
#   3. Bolivia_103757 cross-model disagreement (slide 11 example)
#
# Submit:
#     sbatch slurm/render_figures.sbatch
#
# Then rsync results/figures/paper_*.png back locally.
# --------------------------------------------------------------------------

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE
cd $PROJECT_DIR

module load cuda/12.1.1-binary
eval "$(conda shell.bash hook)"
conda activate floodseg

echo "Job $SLURM_JOB_NAME on $SLURMD_NODENAME at $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

python scripts/render_paper_figures.py

echo "Finished at $(date)"
