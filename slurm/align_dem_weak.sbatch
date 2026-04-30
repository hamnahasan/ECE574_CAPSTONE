#!/bin/bash
#SBATCH --job-name=dem_weak
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus              # CPU only — DEM alignment is I/O + CPU bound
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# --------------------------------------------------------------------------
# Align Copernicus GLO-30 DEM to ~4 385 weakly-labeled S1 chips. This
# unlocks the TriModal weak-pretraining experiment (paper experiment E3).
#
# Reuses scripts/download_dem.py with the new s1_token/dem_token/dem_subdir
# args. Hand-labeled DEM tiles are already cached, so the tile cache hit
# rate should be very high — most "downloads" are no-ops.
# --------------------------------------------------------------------------

set -euo pipefail

WEAK_ROOT=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/WeaklyLabeled
WEAK_S1=$WEAK_ROOT/S1Weak
WEAK_SPLITS=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_weakly_labeled
TILE_CACHE=/lustre/isaac24/scratch/$USER/sen1flood1/v1.1/dem_tiles
PROJECT_DIR=/lustre/isaac24/scratch/$USER/ECE574_CAPSTONE

cd "$PROJECT_DIR"
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate floodseg

# Path verification (the weakly-labeled split directory may live under a
# different name on Isaac — verify before submitting)
fail=0
for path in "$WEAK_S1" "$WEAK_SPLITS"; do
    if [[ ! -e "$path" ]]; then
        echo "MISSING: $path"; fail=1
    fi
done
if [[ $fail -ne 0 ]]; then
    echo "Adjust WEAK_S1 / WEAK_SPLITS in this script and resubmit."
    exit 1
fi

echo "Aligning DEM for weakly-labeled chips ($(date))"

python scripts/download_dem.py \
    --s1_dir     "$WEAK_S1" \
    --splits_dir "$WEAK_SPLITS" \
    --output_dir "$WEAK_ROOT" \
    --tile_cache "$TILE_CACHE" \
    --s1_token   S1Weak \
    --dem_token  DEMWeak \
    --dem_subdir DEMWeak

echo "Finished at $(date)"
