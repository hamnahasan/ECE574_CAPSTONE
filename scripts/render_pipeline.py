"""Generate the end-to-end pipeline figure for the paper.

Saves results/figures/pipeline.png — a single composite diagram covering:

  Data ingestion (Sen1Floods11 + Copernicus GLO-30 DEM)
      |
  Preprocessing (per-modality normalization, DEM bilinear resample 30m -> 10m,
                 augmentation: flip / rotate / random crop)
      |
  Three independent ResNet34 encoders (s1, s2, dem)
      |
  3-way cross-attention bridge at 4 decoder scales (s/8 .. s/64)
                + modality dropout (training only, p sweep)
      |
  Shared U-Net-style decoder
      |
  Inference modes:
      - deterministic single-pass    -> water mask
      - MC Dropout x N=20            -> mean prob + uncertainty + ECE

Use:
    python scripts/render_pipeline.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Visual styling --------------------------------

COLORS = {
    "data":    "#dbe7f3",   # light blue   - data sources
    "prep":    "#f5e1c8",   # light orange - preprocessing
    "encoder": "#cfe5d1",   # light green  - encoders
    "fuse":    "#f1cdd4",   # light pink   - fusion / attention
    "decode":  "#e6dcf2",   # light purple - decoder
    "output":  "#fff3b0",   # light yellow - outputs
}
EDGE = "#333333"


def add_box(ax, x, y, w, h, text, fill, fontsize=9, weight="normal"):
    """Add a rounded rectangle with centered multi-line text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor=EDGE, facecolor=fill, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight, zorder=3)


def add_arrow(ax, x0, y0, x1, y1, label=None, label_offset=(0, 0)):
    """Add an arrow between two points; optional small label near midpoint."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=EDGE, lw=1.4),
        zorder=1,
    )
    if label:
        mx, my = (x0 + x1) / 2 + label_offset[0], (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7,
                style="italic", color="#555555")


def render(save_path):
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Row 1: data sources -----------------------------------------------
    add_box(ax, 0.3, 7.6, 3.4, 1.1,
            "Sen1Floods11 (446 hand-labeled)\nS1 SAR (10 m, VV+VH)\nS2 (10 m, 13 bands, prepped)\nLabels (water / non-water)",
            COLORS["data"], fontsize=8.5)

    add_box(ax, 4.3, 7.6, 3.4, 1.1,
            "Copernicus GLO-30 DEM (30 m)\nelevation + slope (computed)\n-> bilinear resample to 10 m chip grid",
            COLORS["data"], fontsize=8.5)

    add_box(ax, 8.3, 7.6, 4.4, 1.1,
            "Splits: flood_train / valid / test\n+ Bolivia held-out (cross-region)\n+ 4 385 weakly-labeled chips (paper extension)",
            COLORS["data"], fontsize=8.5)

    # ---- Row 2: preprocessing ---------------------------------------------
    add_box(ax, 1.0, 6.0, 5.2, 0.9,
            "Per-modality normalization (means/stds)\nrandom crop 256x256 + h/v flip + rotate 90/180/270",
            COLORS["prep"], fontsize=9)

    add_box(ax, 6.8, 6.0, 5.2, 0.9,
            "Modality dropout p (training only)\n- p sweep {0.0, 0.05, 0.10, 0.20, 0.30, 0.50}\n- inference-time robustness",
            COLORS["prep"], fontsize=8.5)

    add_arrow(ax, 2.0, 7.6, 2.0, 6.9)
    add_arrow(ax, 6.0, 7.6, 6.0, 6.9)
    add_arrow(ax, 10.5, 7.6, 9.0, 6.9)

    # ---- Row 3: encoders ---------------------------------------------------
    enc_y, enc_h = 4.4, 1.1
    add_box(ax, 0.6, enc_y, 2.8, enc_h,
            "ResNet34 encoder\n(S1: 2 ch)\nfeatures @ s/8..s/64",
            COLORS["encoder"], fontsize=9)
    add_box(ax, 4.2, enc_y, 2.8, enc_h,
            "ResNet34 encoder\n(S2: 13 ch)\nfeatures @ s/8..s/64",
            COLORS["encoder"], fontsize=9)
    add_box(ax, 7.8, enc_y, 2.8, enc_h,
            "ResNet34 encoder\n(DEM: 2 ch elev + slope)\nfeatures @ s/8..s/64",
            COLORS["encoder"], fontsize=9)

    add_arrow(ax, 3.6, 6.0, 2.0, 5.5)
    add_arrow(ax, 3.6, 6.0, 5.6, 5.5)
    add_arrow(ax, 9.4, 6.0, 9.2, 5.5)

    # ---- Row 4: fusion bridge ----------------------------------------------
    add_box(ax, 1.5, 2.6, 9.0, 1.2,
            "3-way cross-attention bridge  -  applied at 4 decoder scales (s/8, s/16, s/32, s/64)\n"
            "S1 <-> S2,  S1 <-> DEM,  S2 <-> DEM   |   per-scale 1x1 conv fuses 3 attended maps -> 1",
            COLORS["fuse"], fontsize=9)

    # Encoders -> bridge
    add_arrow(ax, 2.0, enc_y, 3.2, 3.8)
    add_arrow(ax, 5.6, enc_y, 6.0, 3.8)
    add_arrow(ax, 9.2, enc_y, 8.8, 3.8)

    # ---- Row 5: decoder + outputs ------------------------------------------
    add_box(ax, 1.5, 1.0, 4.2, 1.2,
            "Shared U-Net decoder\nDecoderBlock x3 + 2x ConvTranspose2d\n+ Dropout2d (p=0.1)  +  Conv1x1 head",
            COLORS["decode"], fontsize=9)

    add_box(ax, 6.4, 1.0, 6.0, 1.2,
            "Outputs:\n-  Water-segmentation mask (single deterministic pass)\n"
            "-  Mean prob + per-pixel uncertainty (MC Dropout, N=20)\n"
            "-  ECE (test 0.0273  /  Bolivia 0.0451)",
            COLORS["output"], fontsize=8.5)

    add_arrow(ax, 6.0, 2.6, 4.0, 2.2)
    add_arrow(ax, 5.7, 1.6, 6.4, 1.6)

    # ---- Sidebar: training / loss callout ---------------------------------
    ax.text(12.85, 4.95,
            "Training\n-----------------\nDiceCE loss\nAdamW (wd=1e-4)\nlr=2e-4 cosine\n100 epochs\nseed=42\nAMP fp16\n(attn fp32)\ngrad clip 1.0",
            fontsize=7.5, ha="right", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", lw=0.7))

    # ---- Title -------------------------------------------------------------
    ax.text(6.5, 8.85,
            "TriModal Flood Segmentation - End-to-End Pipeline",
            ha="center", va="center", fontsize=14, weight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    render(FIG_DIR / "pipeline.png")


if __name__ == "__main__":
    main()
