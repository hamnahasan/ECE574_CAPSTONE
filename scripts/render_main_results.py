"""Regenerate results/figures/main_results.png from the canonical CSV.

Produces ONE clean slide-ready bar chart for slide 6:
  - 4 bars: Otsu, FCN-ResNet50, Fusion U-Net, TriModal U-Net
  - TriModal highlighted (filled, dark border) — it's our headline model
  - Prithvi-EO foundation-model reference line at 0.8046 (dashed)
  - Value labels on each bar
  - Y axis 0..1.0, gridlines

Numbers are read from results/logs/all_results.csv (rebuild with
scripts/compile_results.py if needed) plus the Otsu test IoU which is
in results/logs/otsu_baseline_results.json.

Usage:
    python scripts/render_main_results.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


LOGS_DIR = Path("results/logs")
FIG_DIR  = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Canonical reference for the literature comparison annotation
PRITHVI_TEST_IOU = 0.8046  # IBM/NASA Prithvi-EO-1.0-100M, Sen1Floods11 test


def load_canonical_numbers():
    """Pull the 4 headline test-split numbers from canonical sources."""
    df = pd.read_csv(LOGS_DIR / "all_results.csv")

    # Otsu test IoU is stored in its own JSON, not in the CSV (CSV has
    # split='unknown' for Otsu because the JSON shape predates the
    # parse_filename split convention).
    with open(LOGS_DIR / "otsu_baseline_results.json") as f:
        otsu_test = json.load(f)["aggregates"]["test"]

    # Helper: pull one row from the test split
    def row(model_name):
        sub = df[(df["model"] == model_name) & (df["split"] == "test")]
        if len(sub) == 0:
            raise ValueError(f"No test row for {model_name}")
        return sub.iloc[0]

    fcn  = row("fcn_baseline")
    fus  = row("fusion_unet")
    tri  = row("trimodal_unet")

    return [
        {"name": "Otsu (S1 VH)",
         "short": "Otsu",
         "iou": float(otsu_test["iou"]),
         "dice": float(otsu_test["dice"]),
         "precision": float(otsu_test["precision"]),
         "recall": float(otsu_test["recall"])},
        {"name": "FCN-ResNet50 (S1)",
         "short": "FCN-ResNet50",
         "iou": float(fcn["iou"]),
         "dice": float(fcn["dice"]),
         "precision": float(fcn["precision"]),
         "recall": float(fcn["recall"])},
        {"name": "Fusion U-Net (S1+S2 cross-attn)",
         "short": "Fusion U-Net",
         "iou": float(fus["iou"]),
         "dice": float(fus["dice"]),
         "precision": float(fus["precision"]),
         "recall": float(fus["recall"])},
        {"name": "TriModal U-Net (S1+S2+DEM)",
         "short": "TriModal U-Net",
         "iou": float(tri["iou"]),
         "dice": float(tri["dice"]),
         "precision": float(tri["precision"]),
         "recall": float(tri["recall"])},
    ]


def render_main_results(rows, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [r["short"] for r in rows]
    ious   = [r["iou"]   for r in rows]

    # Standard color progression (red->orange->blue->green) with TriModal
    # highlighted (saturated green + thicker black border)
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
    edges  = ["white", "white", "white", "black"]
    widths = [1.0, 1.0, 1.0, 2.5]

    bars = ax.bar(labels, ious, color=colors, edgecolor=edges,
                  linewidth=widths, width=0.62, zorder=3)

    # Value labels on top of each bar
    for b, v in zip(bars, ious):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", fontweight="bold", fontsize=12, zorder=4)

    # Prithvi reference line — dashed grey + small label
    ax.axhline(PRITHVI_TEST_IOU, color="#666666", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(len(labels) - 0.5, PRITHVI_TEST_IOU + 0.012,
            f"Prithvi-EO foundation model = {PRITHVI_TEST_IOU:.3f}",
            ha="right", fontsize=9, color="#444444", style="italic")

    ax.set_ylabel("Water-class IoU (test set)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Main Results — Sen1Floods11 Test Set", fontsize=14, pad=14)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Highlight the headline gap visually with an annotation arrow
    tri_iou = ious[-1]
    ax.annotate(
        f"Headline: TriModal within {(PRITHVI_TEST_IOU - tri_iou)*100:.1f} IoU pts\n"
        f"of Prithvi — scratch-trained from 446 chips",
        xy=(3, tri_iou),
        xytext=(2.2, tri_iou - 0.18),
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8dc", ec="#aaa", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def print_table(rows):
    """Print a Markdown table identical to what should appear on the slide."""
    print("\n=== Canonical numbers (use these on the slide) ===\n")
    print("| Model | Water IoU | Dice | Precision | Recall |")
    print("|---|---|---|---|---|")
    for r in rows:
        dice = "—" if r["dice"] == 0 else f"{r['dice']:.3f}"
        prec = "—" if r["precision"] == 0 else f"{r['precision']:.3f}"
        rec  = "—" if r["recall"] == 0 else f"{r['recall']:.3f}"
        bold = "**" if r["short"] == "TriModal U-Net" else ""
        print(f"| {bold}{r['name']}{bold} | {bold}{r['iou']:.3f}{bold} | {dice} | {prec} | {rec} |")


def main():
    rows = load_canonical_numbers()
    render_main_results(rows, FIG_DIR / "main_results.png")
    print_table(rows)
    print(f"\nPrithvi-EO reference: {PRITHVI_TEST_IOU:.4f}")
    print(f"TriModal gap to Prithvi: {(PRITHVI_TEST_IOU - rows[-1]['iou']):.4f} IoU points")


if __name__ == "__main__":
    main()
