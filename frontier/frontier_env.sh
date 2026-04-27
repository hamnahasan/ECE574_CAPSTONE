#!/bin/bash
# Shared Frontier paths and module setup.
#
# Fill these in for your teammate's Frontier account/environment.
# SBATCH account IDs cannot be read from this file, so also replace
# PROJECT_ID_HERE inside each .sbatch file.

set -euo pipefail

# Example scratch pattern on OLCF systems.  Change these if your Frontier account uses a
# different path or if the repo/data live somewhere else.
export PROJECT_DIR="/lustre/orion/PROJECT_ID_HERE/scratch/${USER}/ECE574_CAPSTONE"
export DATA_ROOT="/lustre/orion/PROJECT_ID_HERE/scratch/${USER}/sen1flood1/v1.1/data/flood_events/HandLabeled"
export SPLITS_DIR="/lustre/orion/PROJECT_ID_HERE/scratch/${USER}/sen1flood1/v1.1/splits/flood_handlabeled"

# Conda env created by frontier/setup_frontier.sh.
export FRONTIER_ENV="${PROJECT_DIR}/.frontier_envs/floodseg"

# Frontier / ROCm modules. If OLCF updates module names, run `module avail`
# and adjust these lines.
module purge
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Recommended by OLCF when using non-default CPE modules.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# Keep MIOpen cache on node-local temporary storage to avoid network FS issues.
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID:-interactive}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export PYTHONUNBUFFERED=1

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: PROJECT_DIR does not exist: $PROJECT_DIR"
    echo "Edit frontier/frontier_env.sh for the Frontier account/path."
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: DATA_ROOT does not exist: $DATA_ROOT"
    echo "Edit frontier/frontier_env.sh for the Frontier data path."
    exit 1
fi

if [ ! -d "$FRONTIER_ENV" ]; then
    echo "ERROR: Frontier conda env not found: $FRONTIER_ENV"
    echo "Run: bash frontier/setup_frontier.sh"
    exit 1
fi

conda activate "$FRONTIER_ENV"

