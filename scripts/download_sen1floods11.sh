#!/usr/bin/env bash
# Download Sen1Floods11 dataset from Google Cloud Storage
#
# Prerequisites:
#   pip install gsutil   (or install Google Cloud SDK)
#
# The dataset is hosted at:
#   gs://sen1floods11
#
# Structure after download:
#   data/raw/
#   ├── S1Hand/          # Sentinel-1 hand-labeled chips
#   ├── S2Hand/          # Sentinel-2 hand-labeled chips
#   ├── LabelHand/       # Binary water masks (hand-labeled)
#   ├── S1Weak/          # Sentinel-1 weakly-labeled chips
#   ├── S2Weak/          # Sentinel-2 weakly-labeled chips
#   └── LabelWeak/       # Binary water masks (weak labels)
#
# Usage:
#   chmod +x scripts/download_sen1floods11.sh
#   ./scripts/download_sen1floods11.sh

set -e

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

echo "=== Downloading Sen1Floods11 ==="
echo "Source: gs://sen1floods11"
echo "Target: $RAW_DIR/"
echo ""

# Hand-labeled data (primary — used for train/val/test)
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand "$RAW_DIR/"
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand "$RAW_DIR/"
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand "$RAW_DIR/"

# Weakly-labeled data (for semi-supervised learning)
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak "$RAW_DIR/"
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2Weak "$RAW_DIR/"
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/LabelWeak "$RAW_DIR/"

# Split definitions
gsutil -m cp -r gs://sen1floods11/v1.1/splits "$RAW_DIR/"

echo ""
echo "=== Download complete ==="
echo "Next: inspect chips in notebooks/01_explore_data.ipynb"
