"""Download and preprocess Copernicus DEM GLO-30 for Sen1Floods11 chips.

Pipeline:
  1. Scan all chip bounding boxes from split CSVs
  2. Identify required 1°×1° DEM tiles (Copernicus GLO-30 on AWS S3)
  3. Download missing tiles to a local cache directory
  4. For each chip: crop DEM to chip extent, resample to 512×512, compute slope
  5. Save 2-band GeoTIFF per chip: band 1 = elevation (m), band 2 = slope (degrees)

Output directory structure mirrors S1Hand/:
  <output_dir>/DEMHand/<Country>_<id>_DEMHand.tif

Copernicus DEM GLO-30 is freely available on AWS:
  https://copernicus-dem-30m.s3.amazonaws.com/

Usage:
    python scripts/download_dem.py \
        --s1_dir F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled/S1Hand \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
        --output_dir F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --tile_cache F:/Sen1Flood1/v1.1/dem_tiles
"""

import argparse
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, calculate_default_transform
from tqdm import tqdm


# Copernicus DEM GLO-30 public HTTPS base URL (no auth required)
DEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat_floor, lon_floor):
    """Return the DEM tile filename stem for a given 1°×1° tile origin."""
    lat_pfx = "N" if lat_floor >= 0 else "S"
    lon_pfx = "E" if lon_floor >= 0 else "W"
    lat_str = f"{abs(lat_floor):02d}"
    lon_str = f"{abs(lon_floor):03d}"
    return f"Copernicus_DSM_COG_10_{lat_pfx}{lat_str}_00_{lon_pfx}{lon_str}_00_DEM"


def tile_url(stem):
    return f"{DEM_BASE_URL}/{stem}/{stem}.tif"


def tiles_for_bounds(left, bottom, right, top):
    """Return set of (lat_floor, lon_floor) tiles covering the bounding box."""
    tiles = set()
    for lat in range(math.floor(bottom), math.ceil(top)):
        for lon in range(math.floor(left), math.ceil(right)):
            tiles.add((lat, lon))
    return tiles


def download_tile(stem, tile_cache):
    """Download a DEM tile to cache if not already present. Returns local path."""
    local = Path(tile_cache) / f"{stem}.tif"
    if local.exists():
        return local

    url = tile_url(stem)
    try:
        urllib.request.urlretrieve(url, local)
        return local
    except Exception as e:
        print(f"  WARNING: could not download {url}: {e}")
        return None


def compute_slope(elevation, transform):
    """Compute slope in degrees from elevation array.

    Uses central differences with pixel spacing derived from the transform.
    Works in geographic coordinates (degrees) so converts to meters using
    approximate lat/lon → meters conversion.
    """
    # Pixel size in degrees
    dx_deg = abs(transform.a)
    dy_deg = abs(transform.e)

    # Convert to approximate meters (at equator, 1° ≈ 111320m)
    dx_m = dx_deg * 111320
    dy_m = dy_deg * 111320

    # Gradient using central differences (pad edges)
    grad_y, grad_x = np.gradient(elevation.astype(np.float32), dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg.astype(np.float32)


def process_chip(s1_path, output_dir, tile_cache,
                  s1_token="S1Hand", dem_token="DEMHand", dem_subdir="DEMHand"):
    """Crop, resample and save DEM+slope for one chip.

    Args:
        s1_path: Path to S1 .tif (used to get bounds and grid).
        output_dir: Directory to save DEM chips (subdir below).
        tile_cache: Directory with downloaded DEM tiles.
        s1_token, dem_token: Filename tokens to swap (e.g. "S1Hand"→"DEMHand"
            for hand-labeled, "S1Weak"→"DEMWeak" for weakly-labeled).
        dem_subdir: Subdirectory under output_dir to write the DEM chips.

    Returns:
        Path to output chip, or None on failure.
    """
    s1_path = Path(s1_path)
    chip_name = s1_path.stem.replace(s1_token, dem_token)
    out_path = Path(output_dir) / dem_subdir / f"{chip_name}.tif"

    if out_path.exists():
        return out_path  # already done

    # Get chip grid
    with rasterio.open(s1_path) as src:
        chip_crs = src.crs
        chip_transform = src.transform
        chip_bounds = src.bounds
        chip_height = src.height
        chip_width = src.width

    left, bottom, right, top = (
        chip_bounds.left, chip_bounds.bottom,
        chip_bounds.right, chip_bounds.top,
    )

    # Find required tiles
    needed_tiles = tiles_for_bounds(left, bottom, right, top)

    # Collect DEM data from tiles (a chip is small, usually one tile is enough)
    # Use a slightly expanded read window to avoid edge artefacts
    pad = 0.01  # degrees padding
    read_bounds = (left - pad, bottom - pad, right + pad, top + pad)

    dem_arrays = []
    dem_transforms = []
    dem_crs = None

    for lat_f, lon_f in needed_tiles:
        stem = tile_name(lat_f, lon_f)
        tile_path = Path(tile_cache) / f"{stem}.tif"
        if not tile_path.exists():
            continue

        with rasterio.open(tile_path) as src:
            dem_crs = src.crs
            # Read only the window we need
            from rasterio.windows import from_bounds as window_from_bounds
            win = window_from_bounds(
                read_bounds[0], read_bounds[1], read_bounds[2], read_bounds[3],
                src.transform,
            )
            # Clamp to valid
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                continue
            data = src.read(1, window=win).astype(np.float32)
            win_transform = src.window_transform(win)
            dem_arrays.append(data)
            dem_transforms.append(win_transform)

    if not dem_arrays:
        return None

    # Merge tiles if multiple (simple paste — tiles don't overlap for GLO-30)
    if len(dem_arrays) == 1:
        merged = dem_arrays[0]
        merged_transform = dem_transforms[0]
    else:
        from rasterio.merge import merge as rio_merge
        import tempfile, os
        tmp_files = []
        tmp_paths = []
        for arr, t in zip(dem_arrays, dem_transforms):
            p = tempfile.mktemp(suffix=".tif")
            tmp_paths.append(p)
            with rasterio.open(
                p, "w", driver="GTiff", count=1, dtype=arr.dtype,
                crs=dem_crs, transform=t, width=arr.shape[1], height=arr.shape[0],
            ) as dst:
                dst.write(arr[np.newaxis])
        tmp_files = [rasterio.open(p) for p in tmp_paths]
        merged_arr, merged_transform = rio_merge(tmp_files)
        merged = merged_arr[0]
        for f in tmp_files: f.close()
        for p in tmp_paths: os.remove(p)

    # Reproject+resample to chip grid (EPSG:4326, 512×512, exact bounds)
    out_transform = from_bounds(left, bottom, right, top, chip_width, chip_height)
    elevation = np.zeros((chip_height, chip_width), dtype=np.float32)

    reproject(
        source=merged,
        destination=elevation,
        src_transform=merged_transform,
        src_crs=dem_crs,
        dst_transform=out_transform,
        dst_crs=chip_crs,
        resampling=Resampling.bilinear,
    )

    # Fill nodata (ocean/missing tiles) with 0
    elevation = np.nan_to_num(elevation, nan=0.0)

    # Compute slope from the resampled elevation
    slope = compute_slope(elevation, out_transform)

    # Save 2-band GeoTIFF: band1=elevation, band2=slope
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        count=2,
        dtype=np.float32,
        crs=chip_crs,
        transform=out_transform,
        width=chip_width,
        height=chip_height,
        compress="lzw",
    ) as dst:
        dst.write(elevation, 1)
        dst.write(slope, 2)
        dst.update_tags(
            band1="elevation_m",
            band2="slope_degrees",
            source="Copernicus_DEM_GLO-30",
        )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Copernicus DEM GLO-30")
    parser.add_argument("--s1_dir", required=True, help="Path to S1 chip directory (S1Hand or S1Weak)")
    parser.add_argument("--splits_dir", required=True, help="Path to splits dir with flood_*.csv")
    parser.add_argument("--output_dir", required=True, help="Root dir under which dem_subdir/ is created")
    parser.add_argument("--tile_cache", required=True, help="Directory to cache raw DEM tiles")
    # Tokens / subdir let the same script handle hand-labeled and weakly-labeled
    # chips. Defaults preserve the original hand-labeled behavior.
    parser.add_argument("--s1_token",   default="S1Hand",
                        help="S1 filename token; 'S1Hand' or 'S1Weak'")
    parser.add_argument("--dem_token",  default="DEMHand",
                        help="DEM filename token; 'DEMHand' or 'DEMWeak'")
    parser.add_argument("--dem_subdir", default="DEMHand",
                        help="Subdir under --output_dir for produced DEM chips")
    args = parser.parse_args()

    s1_dir = Path(args.s1_dir)
    splits_dir = Path(args.splits_dir)
    tile_cache = Path(args.tile_cache)
    tile_cache.mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / args.dem_subdir).mkdir(parents=True, exist_ok=True)

    # Collect all unique S1 files
    all_s1_files = set()
    for csv in splits_dir.glob("flood_*.csv"):
        df = pd.read_csv(csv, header=None, names=["s1", "lbl"])
        all_s1_files.update(df["s1"].tolist())
    all_s1_files = [f for f in all_s1_files if not f.startswith(".")]
    print(f"Total chips: {len(all_s1_files)}")

    # Step 1: find all required DEM tiles
    print("\nStep 1: Scanning chip bounds to identify required DEM tiles...")
    required_tiles = set()
    for s1_file in tqdm(all_s1_files):
        s1_path = s1_dir / s1_file
        if not s1_path.exists():
            continue
        with rasterio.open(s1_path) as src:
            b = src.bounds
        required_tiles.update(tiles_for_bounds(b.left, b.bottom, b.right, b.top))

    print(f"Required DEM tiles: {len(required_tiles)}")

    # Step 2: download missing tiles
    print("\nStep 2: Downloading DEM tiles (skipping cached)...")
    failed_tiles = []
    for lat_f, lon_f in tqdm(sorted(required_tiles)):
        stem = tile_name(lat_f, lon_f)
        local = tile_cache / f"{stem}.tif"
        if local.exists():
            continue
        url = tile_url(stem)
        try:
            urllib.request.urlretrieve(url, local)
        except Exception as e:
            failed_tiles.append(stem)
            print(f"  FAILED: {stem} — {e}")

    downloaded = len(required_tiles) - len(failed_tiles)
    print(f"Downloaded: {downloaded}/{len(required_tiles)} tiles")
    if failed_tiles:
        print(f"Failed tiles (ocean/no coverage): {len(failed_tiles)}")

    # Step 3: process each chip
    print("\nStep 3: Cropping + resampling DEM for each chip...")
    success = 0
    for s1_file in tqdm(all_s1_files):
        s1_path = s1_dir / s1_file
        if not s1_path.exists():
            continue
        result = process_chip(s1_path, args.output_dir, tile_cache,
                               s1_token=args.s1_token,
                               dem_token=args.dem_token,
                               dem_subdir=args.dem_subdir)
        if result:
            success += 1

    print(f"\nDone. DEM chips saved: {success}/{len(all_s1_files)}")
    print(f"Output: {Path(args.output_dir) / args.dem_subdir}")
    print(f"Format: 2-band float32 GeoTIFF (band1=elevation_m, band2=slope_deg)")


if __name__ == "__main__":
    main()
