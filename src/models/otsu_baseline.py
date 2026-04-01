"""Otsu thresholding baseline for flood detection on S1 VH band.

This is the classical (non-ML) baseline from the Sen1Floods11 paper.
Otsu's method finds the threshold that minimizes intra-class variance
on the VH backscatter histogram, then classifies pixels below the
threshold as water.
"""

import numpy as np
import rasterio
from pathlib import Path
from skimage.filters import threshold_otsu


def otsu_predict(s1_path, band=1):
    """Run Otsu thresholding on a single S1 chip.

    Args:
        s1_path: Path to S1 GeoTIFF file.
        band: Which band to threshold (0=VV, 1=VH). Default VH.

    Returns:
        pred: (H, W) binary prediction (1=water, 0=non-water).
        threshold: The Otsu threshold value.
    """
    with rasterio.open(s1_path) as src:
        data = src.read(band + 1).astype(np.float32)  # rasterio is 1-indexed

    data = np.nan_to_num(data, nan=0.0)

    # Only compute threshold on valid (non-zero) pixels
    valid_mask = data != 0
    valid_pixels = data[valid_mask]

    if len(valid_pixels) == 0:
        return np.zeros_like(data, dtype=np.int64), 0.0

    threshold = threshold_otsu(valid_pixels)

    # Water = below threshold (low backscatter), only where valid
    pred = np.zeros_like(data, dtype=np.int64)
    pred[(data < threshold) & valid_mask] = 1

    return pred, threshold


def evaluate_otsu_on_split(s1_dir, label_dir, split_csv, band=1):
    """Run Otsu on all chips in a split and compute metrics.

    Args:
        s1_dir: Directory with S1 .tif files.
        label_dir: Directory with label .tif files.
        split_csv: CSV file with (s1_file, label_file) rows.
        band: Band index (0=VV, 1=VH).

    Returns:
        results: list of per-chip dicts with metrics.
        aggregate: dict with aggregate metrics over the split.
    """
    import pandas as pd
    from src.utils.metrics import compute_metrics, MetricAccumulator

    s1_dir = Path(s1_dir)
    label_dir = Path(label_dir)

    df = pd.read_csv(split_csv, header=None, names=["s1_file", "label_file"])
    accumulator = MetricAccumulator()
    results = []

    for _, row in df.iterrows():
        # Predict
        pred, thresh = otsu_predict(s1_dir / row["s1_file"], band=band)

        # Load label
        with rasterio.open(label_dir / row["label_file"]) as src:
            label = src.read(1).astype(np.int64)
        label[label == -1] = 255

        # Per-chip metrics
        metrics = compute_metrics(pred, label, ignore_index=255)
        metrics["file"] = row["s1_file"]
        metrics["threshold"] = thresh
        results.append(metrics)

        # Accumulate
        accumulator.update(pred, label, ignore_index=255)

    aggregate = accumulator.compute()
    return results, aggregate
