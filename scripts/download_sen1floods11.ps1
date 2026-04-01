# Download Sen1Floods11 dataset from Google Cloud Storage
# Prerequisites:
#   pip install gs-chunked-io gsutil (or install Google Cloud SDK)
#
# Usage:
#   .\scripts\download_sen1floods11.ps1

# First, activate the virtual environment if needed
# & ".\venv\Scripts\Activate.ps1"

$RAW_DIR = "data/raw"
New-Item -ItemType Directory -Force -Path $RAW_DIR | Out-Null

Write-Host "=== Downloading Sen1Floods11 ===" 
Write-Host "Source: gs://sen1floods11"
Write-Host "Target: $RAW_DIR/"
Write-Host ""

# Hand-labeled data (primary — used for train/val/test)
Write-Host "Downloading S1Hand..."
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand "$RAW_DIR/"

Write-Host "Downloading S2Hand..."
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand "$RAW_DIR/"

Write-Host "Downloading LabelHand..."
gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand "$RAW_DIR/"

# Split definitions
Write-Host "Downloading splits..."
gsutil -m cp -r gs://sen1floods11/v1.1/splits "$RAW_DIR/"

Write-Host ""
Write-Host "=== Download complete ==="
Write-Host "Next: inspect chips in notebooks/01_explore_data.ipynb"
