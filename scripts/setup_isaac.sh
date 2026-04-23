#!/bin/bash
# -----------------------------------------------------------------------------
# Isaac (UTK ISAAC-NG) environment setup for ECE574 flood segmentation.
#
# Run this ONCE after cloning the repo to /lustre/isaac24/scratch/$USER/.
# Creates a conda env named 'floodseg' with PyTorch + rasterio + all deps.
#
# CRITICAL: Isaac home directory (/nfs/home/$USER) has a small disk quota
# (~10GB). Conda envs and package caches MUST go on lustre scratch
# (/lustre/isaac24/scratch/$USER) which has TBs of space.
#
# This script redirects:
#   - conda env directory          -> $LUSTRE/conda_envs/
#   - conda package cache          -> $LUSTRE/conda_pkgs/
#   - pip cache                    -> $LUSTRE/pip_cache/
#   - .cache (notices, etc)        -> $LUSTRE/.cache/
#
# Usage:
#   bash scripts/setup_isaac.sh
# -----------------------------------------------------------------------------

set -euo pipefail

ENV_NAME=floodseg
LUSTRE_DIR="/lustre/isaac24/scratch/$USER"

# Sanity: lustre scratch must exist
if [ ! -d "$LUSTRE_DIR" ]; then
    echo "ERROR: $LUSTRE_DIR does not exist. Are you on Isaac?"
    exit 1
fi

# Redirect ALL conda/pip storage to lustre to avoid home quota errors
ENVS_DIR="$LUSTRE_DIR/conda_envs"
PKGS_DIR="$LUSTRE_DIR/conda_pkgs"
PIP_CACHE="$LUSTRE_DIR/pip_cache"
XDG_CACHE="$LUSTRE_DIR/.cache"
mkdir -p "$ENVS_DIR" "$PKGS_DIR" "$PIP_CACHE" "$XDG_CACHE"

# Conda config: tell conda to use lustre paths.
# These are persistent (written to ~/.condarc) so future logins inherit them.
conda config --add envs_dirs "$ENVS_DIR"
conda config --add pkgs_dirs "$PKGS_DIR"

# Environment vars for this session and any subprocess
export PIP_CACHE_DIR="$PIP_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE"
export CONDA_ENVS_PATH="$ENVS_DIR"
export CONDA_PKGS_DIRS="$PKGS_DIR"

echo "=== Storage paths ==="
echo "  Envs:      $ENVS_DIR"
echo "  Pkgs:      $PKGS_DIR"
echo "  Pip cache: $PIP_CACHE"
echo "  XDG cache: $XDG_CACHE"

echo "=== Loading Isaac modules ==="
# Anaconda is already on PATH from the user's default profile (base env active),
# so we do NOT load an anaconda module. Just load CUDA.
# Available CUDA versions (as of 2026): 11.8.0-binary, 12.1.1-binary, 12.2.0-binary,
# 12.6.3-binary, 12.9.1-binary. We pin 12.1.1 to match the PyTorch wheels.
module load cuda/12.1.1-binary

# Check if env already exists (in either default or lustre location)
if conda env list | grep -q "${ENV_NAME}"; then
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
echo "    module load cuda/12.1.1-binary"
echo "    source activate $ENV_NAME"
echo ""
echo "Storage paths persisted to ~/.condarc so future logins use lustre automatically."
echo "If you also want pip cache/XDG cache persistent across sessions, add to ~/.bashrc:"
echo "    export PIP_CACHE_DIR=$PIP_CACHE"
echo "    export XDG_CACHE_HOME=$XDG_CACHE"
