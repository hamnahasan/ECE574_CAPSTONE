#!/usr/bin/env python3
"""Download Sen1Floods11 dataset from Google Cloud Storage."""

import os
import subprocess
from pathlib import Path

def download_dataset():
    """Download Sen1Floods11 dataset using gsutil."""
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Downloading Sen1Floods11 ===")
    print("Source: gs://sen1floods11")
    print(f"Target: {raw_dir}/")
    print()
    
    # List of items to download
    downloads = [
        ("S1Hand", "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand"),
        ("S2Hand", "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand"),
        ("LabelHand", "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"),
        ("splits", "gs://sen1floods11/v1.1/splits"),
    ]
    
    for name, source in downloads:
        try:
            print(f"Downloading {name}...")
            result = subprocess.run(
                ["gsutil", "-m", "cp", "-r", source, str(raw_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ {name} downloaded successfully")
        except FileNotFoundError:
            print(f"ERROR: gsutil not found. Please install it via:")
            print("  pip install gs-chunked-io gsutil")
            print(" OR install Google Cloud SDK from:")
            print("  https://cloud.google.com/sdk/docs/install")
            return False
        except subprocess.CalledProcessError as e:
            print(f"ERROR downloading {name}:")
            print(e.stderr)
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    print()
    print("=== Download complete ===")
    print("Next: inspect chips in notebooks/01_explore_data.ipynb")
    return True

if __name__ == "__main__":
    success = download_dataset()
    exit(0 if success else 1)
