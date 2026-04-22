#!/bin/bash
# -----------------------------------------------------------------------------
# Isaac (UTK ISAAC-NG) environment setup for ECE574 flood segmentation.
#
# Run this ONCE after cloning the repo to /lustre/isaac24/scratch/$USER/.
# Creates a conda env named 'floodseg' with PyTorch + rasterio + all deps.
#
# Usage:
#   bash scripts/setup_isaac.sh
# -----------------------------------------------------------------------------

set -euo pipefail

ENV_NAME=floodseg

echo "=== Loading Isaac modules ==="
module load anaconda3/2024.02
module load cuda/12.1

# Check if env already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Env '$ENV_NAME' already exists. Skipping creation."
    echo "To recreate: conda env remove -n $ENV_NAME"
else
    echo "=== Creating conda env: $ENV_NAME ==="
    conda create -n $ENV_NAME python=3.10 -y
fi

source activate $ENV_NAME

echo "=== Installing PyTorch (CUDA 12.1) ==="
# Pinning to CUDA 12.1 to match the module; if Isaac updates, adjust --index-url
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing project dependencies ==="
pip install \
    numpy \
    pandas \
    matplotlib \
    rasterio \
    scikit-image \
    scikit-learn \
    pyyaml \
    tqdm \
    tensorboard

echo "=== Verifying install ==="
python - <<'PY'
import torch, rasterio, numpy, pandas, matplotlib
print(f"PyTorch:     {torch.__version__}")
print(f"CUDA avail:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"CUDA ver:    {torch.version.cuda}")
print(f"rasterio:    {rasterio.__version__}")
print(f"numpy:       {numpy.__version__}")
print(f"pandas:      {pandas.__version__}")
PY

echo ""
echo "=== Setup complete. Activate the env with: ==="
echo "    module load anaconda3/2024.02 cuda/12.1"
echo "    source activate $ENV_NAME"
