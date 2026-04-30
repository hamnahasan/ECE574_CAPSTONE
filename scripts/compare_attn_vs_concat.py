"""Build the matched cross-attention-vs-concatenation comparison table.

Once the modality-dropout sensitivity sweep finishes, the run named
``trimodal_p000`` is a TriModal U-Net trained at p=0 — i.e. without modality
dropout, identical seed/lr/epochs/batch size to the EarlyFusionUNet variant
``ablation_s1_s2_dem``. That gives us a fully controlled experiment:

  Model                              Mod dropout   Fusion strategy
  ---------------------------------- ------------- -------------------------
  ablation_s1_s2_dem (EarlyFusion)   none          concat at input
  trimodal_p000      (TriModal)      none          3-way cross-attn x 4 scales

Same data, same splits, same hyperparameters. The only thing that differs
between them is fusion strategy. This script pulls the test metrics for
both runs from ``results/logs/`` and writes:

  results/logs/attn_vs_concat_matched.json
  results/figures/attn_vs_concat_matched.png

Run after both checkpoints exist.

Usage:
    python scripts/compare_attn_vs_concat.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


LOGS_DIR = Path("results/logs")
FIG_DIR  = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Source files for the two test-set evaluations. Both are flat metric dicts
# with iou / dice / precision / recall / accuracy.
SOURCES = {
    "concat (EarlyFusion)":     LOGS_DIR / "ablation_s1_s2_dem_test_results.json",
    "cross-attn (TriModal p=0)": LOGS_DIR / "trimodal_p000_test_results.json",
}


def load_metrics(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}.\n"
            "Run the sensitivity sweep + the s1_s2_dem ablation before "
            "calling this script."
        )
    with open(path) as f:
        return json.load(f)


def main():
    rows = {}
    for label, path in SOURCES.items():
        m = load_metrics(path)
        rows[label] = {
            "iou":       float(m["iou"]),
            "dice":      float(m["dice"]),
            "precision": float(m["precision"]),
            "recall":    float(m["recall"]),
            "accuracy":  float(m["accuracy"]),
            "source":    str(path),
        }

    # JSON output ----------------------------------------------------------
    out_json = LOGS_DIR / "attn_vs_concat_matched.json"
    with open(out_json, "w") as f:
        json.dump({
            "experiment": "Matched cross-attention vs concatenation (s1+s2+dem)",
            "controls": {
                "modalities":    "s1+s2+dem",
                "seed":          42,
                "epochs":        100,
                "lr":            2e-4,
                "batch_size":    4,
                "crop_size":     256,
                "modality_dropout": 0.0,
            },
            "rows": rows,
        }, f, indent=2)
    print(f"Saved: {out_json}")

    # Markdown table ------------------------------------------------------
    print("\nMatched experiment (s1+s2+dem, no modality dropout, same seed/epochs):\n")
    print("| Model | IoU | Dice | Precision | Recall |")
    print("|---|---|---|---|---|")
    for label, m in rows.items():
        print(f"| {label} | {m['iou']:.4f} | {m['dice']:.4f} | "
              f"{m['precision']:.4f} | {m['recall']:.4f} |")

    iou_gap = rows["cross-attn (TriModal p=0)"]["iou"] - rows["concat (EarlyFusion)"]["iou"]
    print(f"\nDelta IoU (cross-attn - concat): {iou_gap:+.4f}")

    # Bar chart -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(rows.keys())
    ious   = [rows[l]["iou"] for l in labels]

    colors = ["#5bc0de", "#5cb85c"]
    bars = ax.bar(labels, ious, color=colors, edgecolor="black",
                  linewidth=1.0, width=0.55, zorder=3)
    for b, v in zip(bars, ious):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", fontweight="bold", fontsize=11, zorder=4)

    ax.set_ylabel("Water-class IoU (test)", fontsize=11)
    ax.set_ylim(0, max(ious) * 1.18)
    ax.set_title(
        "Matched comparison: fusion strategy at s1+s2+dem\n"
        "(same seed, epochs, lr, batch — modality dropout = 0 for both)",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    out_png = FIG_DIR / "attn_vs_concat_matched.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
