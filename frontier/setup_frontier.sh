#!/bin/bash
# One-time Frontier environment setup for the flood segmentation project.
#
# Run from the repo root on Frontier:
#   bash frontier/setup_frontier.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
ENV_DIR="${FRONTIER_ENV:-${PROJECT_DIR}/.frontier_envs/floodseg}"
CACHE_DIR="${PROJECT_DIR}/.frontier_cache"

mkdir -p "$(dirname "$ENV_DIR")" "$CACHE_DIR/pip" "$CACHE_DIR/conda" "$CACHE_DIR/xdg"

module purge
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"

echo "Project: $PROJECT_DIR"
echo "Env:     $ENV_DIR"
echo "Cache:   $CACHE_DIR"

if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" python=3.10 -y
else
    echo "Conda env already exists: $ENV_DIR"
fi

conda activate "$ENV_DIR"

python -m pip install --upgrade pip

# ROCm PyTorch wheels. If OLCF changes the ROCm module version, update both
# the module in frontier_env.sh and this index URL.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0

python -m pip install -r requirements.txt
python -m pip install scikit-learn tensorboard

python - <<'PY'
import torch
import rasterio
print(f"PyTorch:    {torch.__version__}")
print(f"HIP/ROCm:   {torch.version.hip}")
print(f"GPU avail:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count:  {torch.cuda.device_count()}")
    print(f"GPU 0:      {torch.cuda.get_device_name(0)}")
print(f"rasterio:   {rasterio.__version__}")
PY

echo ""
echo "Setup complete."
echo "Remember to edit frontier/frontier_env.sh before submitting jobs."

