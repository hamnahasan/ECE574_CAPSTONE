"""Helper: append DEM exploration cells to 01_explore_data.ipynb."""
import json
from pathlib import Path

nb_path = Path("notebooks/01_explore_data.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

# Remove any previously added DEM cells to stay idempotent
nb["cells"] = [c for c in nb["cells"] if "DEM" not in "".join(c["source"])[:60]]

def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }

dem_cells = [
    md_cell([
        "## 10. DEM Exploration (Elevation + Slope)\n",
        "\n",
        "Copernicus GLO-30 DEM chips aligned to the S1/S2 grid — 2-band GeoTIFF:\n",
        "- **Band 1:** Elevation in meters\n",
        "- **Band 2:** Slope in degrees\n",
        "\n",
        "Water accumulates in low-elevation, flat areas. "
        "Elevation + slope are physics-informed features that don't depend on clouds or SAR signal quality.",
    ]),
    code_cell([
        "DEM_DIR = DATA_ROOT / 'data' / 'flood_events' / 'HandLabeled' / 'DEMHand'\n",
        "dem_files = [f for f in sorted(DEM_DIR.glob('*.tif')) if not f.name.startswith('.')]\n",
        "print(f'DEM chips found: {len(dem_files)}')\n",
        "\n",
        "# Inspect one chip — use the same sample as earlier sections\n",
        "dem_file = DEM_DIR / sample_row['s1_file'].replace('S1Hand', 'DEMHand')\n",
        "with rasterio.open(dem_file) as src:\n",
        "    elevation = src.read(1).astype(np.float32)\n",
        "    slope     = src.read(2).astype(np.float32)\n",
        "    print(f'Shape: {src.width}x{src.height}, CRS: {src.crs}')\n",
        "\n",
        "print(f'Elevation — min: {elevation.min():.1f} m, max: {elevation.max():.1f} m, mean: {elevation.mean():.1f} m')\n",
        "print(f'Slope     — min: {slope.min():.2f} deg, max: {slope.max():.2f} deg, mean: {slope.mean():.2f} deg')",
    ]),
    code_cell([
        "# Side-by-side: S1 VH | Elevation | Slope | Label\n",
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n",
        "\n",
        "vh = s1[1]\n",
        "axes[0].imshow(vh, cmap='gray', vmin=np.percentile(vh,2), vmax=np.percentile(vh,98))\n",
        "axes[0].set_title('S1 VH (SAR)'); axes[0].axis('off')\n",
        "\n",
        "im1 = axes[1].imshow(elevation, cmap='terrain')\n",
        "plt.colorbar(im1, ax=axes[1], fraction=0.046, label='m')\n",
        "axes[1].set_title('Elevation (m)'); axes[1].axis('off')\n",
        "\n",
        "im2 = axes[2].imshow(slope, cmap='YlOrRd', vmin=0, vmax=np.percentile(slope, 99))\n",
        "plt.colorbar(im2, ax=axes[2], fraction=0.046, label='deg')\n",
        "axes[2].set_title('Slope (degrees)'); axes[2].axis('off')\n",
        "\n",
        "lbl_disp = np.where(label_data == 255, 0.5, label_data.astype(float))\n",
        "axes[3].imshow(lbl_disp, cmap='Blues', vmin=0, vmax=1)\n",
        "axes[3].set_title('Flood Label'); axes[3].axis('off')\n",
        "\n",
        "chip_id = sample_row['s1_file'].replace('_S1Hand.tif', '')\n",
        "fig.suptitle(f'DEM vs S1 vs Label — {chip_id}', fontsize=13)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../results/figures/dem_vs_s1_label.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()",
    ]),
    code_cell([
        "# Statistics across all chips: elevation distribution + water vs land elevation\n",
        "from tqdm import tqdm\n",
        "\n",
        "LABEL_DIR = DATA_ROOT / 'data' / 'flood_events' / 'HandLabeled' / 'LabelHand'\n",
        "elev_means, slope_means = [], []\n",
        "water_elev, land_elev = [], []\n",
        "\n",
        "for dem_f in tqdm(dem_files, desc='Scanning DEM chips'):\n",
        "    with rasterio.open(dem_f) as src:\n",
        "        elev  = src.read(1).astype(np.float32)\n",
        "        slp   = src.read(2).astype(np.float32)\n",
        "    elev_means.append(float(elev.mean()))\n",
        "    slope_means.append(float(slp.mean()))\n",
        "\n",
        "    lbl_f = LABEL_DIR / dem_f.name.replace('DEMHand', 'LabelHand')\n",
        "    if lbl_f.exists():\n",
        "        with rasterio.open(lbl_f) as src:\n",
        "            lbl = src.read(1)\n",
        "        if (lbl == 1).sum() > 200:\n",
        "            water_elev.append(float(elev[lbl == 1].mean()))\n",
        "        if (lbl == 0).sum() > 200:\n",
        "            land_elev.append(float(elev[lbl == 0].mean()))\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
        "\n",
        "axes[0].hist(elev_means, bins=30, color='steelblue', edgecolor='white')\n",
        "axes[0].set_xlabel('Mean Elevation (m)'); axes[0].set_ylabel('Chip count')\n",
        "axes[0].set_title('Chip Mean Elevation Distribution'); axes[0].grid(alpha=0.3)\n",
        "\n",
        "axes[1].hist(slope_means, bins=30, color='coral', edgecolor='white')\n",
        "axes[1].set_xlabel('Mean Slope (deg)'); axes[1].set_ylabel('Chip count')\n",
        "axes[1].set_title('Chip Mean Slope Distribution'); axes[1].grid(alpha=0.3)\n",
        "\n",
        "axes[2].boxplot([water_elev, land_elev], labels=['Water pixels', 'Land pixels'],\n",
        "               patch_artist=True, boxprops=dict(facecolor='#3182bd', alpha=0.7))\n",
        "axes[2].set_ylabel('Mean Elevation (m)')\n",
        "axes[2].set_title('Elevation: Water vs Land Pixels'); axes[2].grid(alpha=0.3, axis='y')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('../results/figures/dem_statistics.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f'Chips: {len(dem_files)}')\n",
        "print(f'Mean elevation: {np.mean(elev_means):.1f} m  |  Mean slope: {np.mean(slope_means):.2f} deg')\n",
        "if water_elev and land_elev:\n",
        "    print(f'Water pixels avg elevation: {np.mean(water_elev):.1f} m')\n",
        "    print(f'Land  pixels avg elevation: {np.mean(land_elev):.1f} m')\n",
        "    diff = np.mean(land_elev) - np.mean(water_elev)\n",
        "    print(f'=> Land is on average {diff:.1f} m higher than water pixels (expected for floods)')",
    ]),
]

nb["cells"].extend(dem_cells)

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
