"""Per-chip error analysis — where do models succeed vs fail?

Reads the per_chip arrays from all *_test_results.json and *_bolivia_results.json
and answers:

  1. Which chips does each model get RIGHT? (top-5 by IoU per model)
  2. Which chips does each model get WRONG? (bottom-5 by IoU per model)
  3. Are there "universally hard" chips — ones where every model fails?
     (These reveal data issues, not model weakness.)
  4. How does performance break down by country/flood event?
     (Extracted from chip names: "Bolivia_290290_S1Hand.tif" -> "Bolivia".)

Figures saved:
  - results/figures/iou_distributions.png       box-plot per model
  - results/figures/iou_by_country.png          per-country mean IoU bar chart
  - results/figures/hard_chips_heatmap.png      per-chip x per-model IoU grid

Markdown report: results/logs/error_analysis.md

Usage:
    python scripts/error_analysis.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOGS_DIR = Path("results/logs")
FIG_DIR  = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Canonical model set for the paper
PRIMARY_MODELS = ["fcn_baseline", "fusion_unet", "trimodal_unet"]
ABLATION_MODELS = [
    "ablation_s1", "ablation_s2", "ablation_dem",
    "ablation_s1_s2", "ablation_s1_dem", "ablation_s2_dem",
    "ablation_s1_s2_dem",
]


# Same aliases as compile_results.py — so Bolivia JSONs resolve to full names
MODEL_NAME_ALIASES = {
    "fcn":      "fcn_baseline",
    "fusion":   "fusion_unet",
    "trimodal": "trimodal_unet",
}


def parse_model_split(stem):
    if stem.endswith("_results"):
        stem = stem[:-len("_results")]
    for split in ["test", "bolivia", "val", "train"]:
        suffix = "_" + split
        if stem.endswith(suffix):
            model = stem[:-len(suffix)]
            model = MODEL_NAME_ALIASES.get(model, model)
            return model, split
    return stem, "unknown"


def country_of(chip_name):
    """Extract country/event code from chip filename.

    'Bolivia_290290_S1Hand.tif' -> 'Bolivia'
    'Sri-Lanka_12345_S1Hand.tif' -> 'Sri-Lanka'
    """
    stem = Path(chip_name).stem
    # Drop any trailing _S1Hand/_S2Hand/_LabelHand/etc.
    stem = re.sub(r"_(S1Hand|S2Hand|LabelHand|DEMHand|SlopeHand).*$", "", stem)
    # Country is everything before the first '_<digits>'
    parts = re.split(r"_\d", stem, maxsplit=1)
    return parts[0] if parts else stem


def load_per_chip(json_path):
    """Return a DataFrame with one row per chip from a *_results.json."""
    with open(json_path) as f:
        data = json.load(f)
    per_chip = data.get("per_chip", [])
    if not per_chip:
        return None
    df = pd.DataFrame(per_chip)
    df["country"] = df["file"].apply(country_of)
    return df


def analyze_split(chip_data, split, md_lines):
    """Per-chip analysis for a given split. Works whenever per_chip data exists.

    Returns the list of primary models that had data for this split.
    """
    md_lines.append(f"## Best / Worst chips per model — {split} split\n")

    available_primaries = []
    for m in PRIMARY_MODELS:
        key = (m, split)
        if key not in chip_data:
            print(f"  [skip] no per-chip data for {m} on {split}")
            continue
        available_primaries.append(m)
        df = chip_data[key].sort_values("iou", ascending=False)
        print(f"=== {m} — {split} ===")
        print(f"  n chips:        {len(df)}")
        print(f"  IoU mean:       {df['iou'].mean():.4f}")
        print(f"  IoU median:     {df['iou'].median():.4f}")
        print(f"  IoU <0.3 count: {(df['iou'] < 0.3).sum()}")
        print()

        md_lines.append(f"### {m}\n")
        md_lines.append(f"- **n chips:** {len(df)}  |  mean IoU: {df['iou'].mean():.4f}  |  median: {df['iou'].median():.4f}\n")
        md_lines.append(f"- **<0.3 IoU (failure):** {(df['iou'] < 0.3).sum()}  |  **>0.9 IoU (near-perfect):** {(df['iou'] > 0.9).sum()}\n\n")

        md_lines.append("**Top-5 (easiest chips):**\n\n")
        md_lines.append("| Chip | IoU | Precision | Recall |\n|---|---|---|---|\n")
        for _, r in df.head(5).iterrows():
            md_lines.append(f"| {r['file']} | {r['iou']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |\n")

        md_lines.append("\n**Bottom-5 (hardest chips):**\n\n")
        md_lines.append("| Chip | IoU | Precision | Recall |\n|---|---|---|---|\n")
        for _, r in df.tail(5).iterrows():
            md_lines.append(f"| {r['file']} | {r['iou']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |\n")
        md_lines.append("\n")

    return available_primaries


def main():
    # ------------------------------------------------------------------
    # 1. Load all per-chip data
    # ------------------------------------------------------------------
    chip_data = {}   # (model, split) -> DataFrame of per-chip metrics
    for jf in sorted(LOGS_DIR.glob("*_results.json")):
        if jf.name.startswith("all_"):
            continue
        model, split = parse_model_split(jf.stem)
        df = load_per_chip(jf)
        if df is not None and len(df) > 0:
            chip_data[(model, split)] = df

    print(f"Loaded per-chip data for {len(chip_data)} (model, split) combos")
    for k in sorted(chip_data.keys()):
        print(f"  - {k[0]} / {k[1]}  ({len(chip_data[k])} chips)")
    print()

    md_lines = ["# Error Analysis — Where do models succeed and fail?\n\n"]

    # ------------------------------------------------------------------
    # 2. Per-chip analysis on whichever splits have data
    # ------------------------------------------------------------------
    splits_with_data = sorted({s for (_, s) in chip_data.keys()})

    if not any((m, "test") in chip_data for m in PRIMARY_MODELS):
        md_lines.append(
            "> **Note:** test-split per-chip data is not available locally. Training scripts\n"
            "> saved flat aggregate dicts rather than `{aggregate, per_chip}`. To enable test\n"
            "> per-chip analysis, re-run on Isaac:\n"
            "> ```\n"
            "> for M in fcn fusion trimodal; do\n"
            ">   python scripts/evaluate.py --model $M \\\n"
            ">     --checkpoint results/checkpoints/${M}_unet_best.pt \\\n"
            ">     --data_root /lustre/.../HandLabeled --splits_dir /lustre/.../flood_handlabeled \\\n"
            ">     --split test --output results/logs/${M}_test_results.json\n"
            "> done\n"
            "> ```\n\n"
        )

    primary_per_split = {}   # split -> list of primary models with data
    for split in splits_with_data:
        primary_per_split[split] = analyze_split(chip_data, split, md_lines)

    # ------------------------------------------------------------------
    # 3. "Hard chips" — low IoU across ALL primary models (any split with ≥2 models)
    # ------------------------------------------------------------------
    best_split = max(splits_with_data, key=lambda s: len(primary_per_split.get(s, [])),
                     default=None)
    primary_test = primary_per_split.get(best_split, []) if best_split else []

    if len(primary_test) >= 2:
        md_lines.append(f"## Cross-model analysis on `{best_split}` split\n\n")
        wide = None
        for m in primary_test:
            df = chip_data[(m, best_split)][["file", "iou"]].rename(columns={"iou": m})
            wide = df if wide is None else wide.merge(df, on="file", how="outer")
        wide["min_iou"]  = wide[primary_test].min(axis=1)
        wide["max_iou"]  = wide[primary_test].max(axis=1)
        wide["spread"]   = wide["max_iou"] - wide["min_iou"]
        wide["country"]  = wide["file"].apply(country_of)

        print(f"=== Universally hard chips on {best_split} (max IoU across all models < 0.3) ===")
        hard = wide[wide["max_iou"] < 0.3].sort_values("max_iou")
        print(f"  count: {len(hard)}\n")
        if len(hard) > 0:
            md_lines.append(f"### Universally hard chips — every primary model fails (max IoU < 0.3)\n\n")
            md_lines.append("These reveal dataset issues, not model weakness. "
                            "Investigate label quality / ambiguous cases.\n\n")
            header = "| Chip | Country | " + " | ".join(primary_test) + " |\n"
            sep    = "|" + "---|" * (2 + len(primary_test)) + "\n"
            md_lines.append(header); md_lines.append(sep)
            for _, r in hard.head(15).iterrows():
                row = f"| {r['file']} | {r['country']} "
                for m in primary_test:
                    row += f"| {r[m]:.3f} " if pd.notna(r[m]) else "| - "
                md_lines.append(row + "|\n")
            md_lines.append("\n")

        print(f"=== High-disagreement chips on {best_split} (IoU spread > 0.3 between models) ===")
        disag = wide[wide["spread"] > 0.3].sort_values("spread", ascending=False)
        print(f"  count: {len(disag)}\n")
        if len(disag) > 0:
            md_lines.append(f"### High-disagreement chips (IoU spread > 0.3 between models)\n\n")
            md_lines.append("Chips where one model succeeds and another fails - "
                            "these are the ones where modality / architecture choices matter.\n\n")
            header = "| Chip | Country | " + " | ".join(primary_test) + " | Spread |\n"
            sep    = "|" + "---|" * (3 + len(primary_test)) + "\n"
            md_lines.append(header); md_lines.append(sep)
            for _, r in disag.head(15).iterrows():
                row = f"| {r['file']} | {r['country']} "
                for m in primary_test:
                    row += f"| {r[m]:.3f} " if pd.notna(r[m]) else "| - "
                row += f"| {r['spread']:.3f} |\n"
                md_lines.append(row)
            md_lines.append("\n")

    # ------------------------------------------------------------------
    # 4. Per-country breakdown (only if >1 country in data)
    # ------------------------------------------------------------------
    country_df = None
    if len(primary_test) > 0:
        # Collect per-country stats for each primary model
        rows = []
        for m in primary_test:
            grp = chip_data[(m, best_split)].groupby("country")["iou"].agg(["mean", "count"])
            grp.columns = [f"{m}_iou", f"{m}_n"]
            rows.append(grp)
        country_df = pd.concat(rows, axis=1) if rows else None

        if country_df is not None and len(country_df) > 1:
            print(f"=== Per-country mean IoU ({best_split} split) ===")
            print(country_df.round(4))
            print()

            md_lines.append(f"## Per-country breakdown ({best_split} split)\n\n")
            md_lines.append("Flood events with low IoU may indicate SAR/optical signal quality issues "
                            "or atypical flood morphology.\n\n")
            iou_cols = [c for c in country_df.columns if c.endswith("_iou")]
            table = country_df.round(4).reset_index()[["country"] + iou_cols]
            md_lines.append("| " + " | ".join(table.columns) + " |\n")
            md_lines.append("|" + "---|" * len(table.columns) + "\n")
            for _, r in table.iterrows():
                md_lines.append("| " + " | ".join(
                    str(r[c]) if isinstance(r[c], str) else f"{r[c]:.4f}" for c in table.columns
                ) + " |\n")
            md_lines.append("\n")
        elif country_df is not None:
            print(f"Only 1 country in {best_split} split ({country_df.index[0]}) - "
                  f"per-country breakdown not meaningful; skipping.")

    # ------------------------------------------------------------------
    # 5. Figure - per-chip IoU distribution (box plot, uses best_split)
    # ------------------------------------------------------------------
    if best_split:
        fig, ax = plt.subplots(figsize=(9, 5))
        box_data, box_labels = [], []
        for m in PRIMARY_MODELS + ABLATION_MODELS:
            key = (m, best_split)
            if key in chip_data:
                box_data.append(chip_data[key]["iou"].dropna().values)
                box_labels.append(m.replace("ablation_", ""))
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, showfliers=True, patch_artist=True)
            for patch, lbl in zip(bp["boxes"], box_labels):
                patch.set_facecolor("#5cb85c" if lbl.startswith("trimodal") else
                                     "#5bc0de" if lbl.startswith("fusion") else
                                     "#f0ad4e" if lbl.startswith("fcn")    else "#cccccc")
            ax.set_ylabel("Per-chip IoU")
            ax.set_title(f"IoU distribution per chip ({best_split} split)")
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out = FIG_DIR / f"iou_distributions_{best_split}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out}")

    # ------------------------------------------------------------------
    # 6. Figure - per-country bar chart (only if >1 country)
    # ------------------------------------------------------------------
    if country_df is not None and len(country_df) > 1:
        iou_cols = [c for c in country_df.columns if c.endswith("_iou")]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(country_df))
        w = 0.8 / len(iou_cols)
        colors = {"fcn_baseline_iou": "#f0ad4e",
                  "fusion_unet_iou":  "#5bc0de",
                  "trimodal_unet_iou":"#5cb85c"}
        for i, col in enumerate(iou_cols):
            ax.bar(x + (i - len(iou_cols)/2) * w + w/2, country_df[col].values,
                   w, label=col.replace("_iou",""), color=colors.get(col, "#888"))
        ax.set_xticks(x); ax.set_xticklabels(country_df.index, rotation=25, ha="right")
        ax.set_ylabel("Mean per-chip IoU")
        ax.set_title(f"Per-country mean IoU - {best_split} split")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        out = FIG_DIR / f"iou_by_country_{best_split}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # ------------------------------------------------------------------
    # 7. Write markdown report
    # ------------------------------------------------------------------
    md_path = LOGS_DIR / "error_analysis.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
