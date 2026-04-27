"""Aggregate all experiment results into a single CSV + Markdown table.

SCIENTIFIC PURPOSE:
    After running all experiments on Isaac, the `results/logs/` directory
    will contain dozens of JSON files from different models and splits.
    This script walks that directory, pulls the key metrics out of each
    JSON, and produces:

      1. results/logs/all_results.csv — one row per experiment
      2. results/logs/all_results.md  — the ablation/comparison table,
         ready to paste into the paper

    This eliminates manual transcription errors and produces consistent
    formatting across the paper's main results table and ablation table.

Usage:
    python scripts/compile_results.py
    python scripts/compile_results.py --logs_dir results/logs
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# Map filename patterns -> (model_label, split) for readable output.
# Any file matching *_results.json, *_test_results.json, or *_bolivia_results.json
# is pulled in automatically.
# Bolivia eval used short aliases (fcn/fusion/trimodal) while training
# scripts use full names (fcn_baseline/fusion_unet/trimodal_unet). Normalize
# so test and Bolivia rows merge correctly in the notebook.
MODEL_NAME_ALIASES = {
    "fcn":      "fcn_baseline",
    "fusion":   "fusion_unet",
    "trimodal": "trimodal_unet",
}


def parse_filename(stem):
    """Infer (model, split) from the result JSON filename.

    Examples:
        fcn_baseline_test_results  -> ("fcn_baseline", "test")
        fusion_unet_bolivia_results -> ("fusion_unet", "bolivia")
        fusion_bolivia_results -> ("fusion_unet", "bolivia")  # alias normalized
        ablation_s1_s2_dem_test_results -> ("ablation_s1_s2_dem", "test")
        trimodal_unet_test_results -> ("trimodal_unet", "test")
        otsu_test_results -> ("otsu", "test")
    """
    # Strip trailing "_results"
    if stem.endswith("_results"):
        stem = stem[:-len("_results")]

    # Last token is the split name
    for split in ["test", "bolivia", "val", "train"]:
        suffix = "_" + split
        if stem.endswith(suffix):
            model = stem[:-len(suffix)]
            model = MODEL_NAME_ALIASES.get(model, model)
            return model, split

    return stem, "unknown"


def load_metrics(json_path):
    """Load metrics dict from a result JSON file.

    Handles two JSON shapes:
    - Direct metric dict: {"iou": 0.7, "dice": 0.8, ...}
    - Nested: {"aggregate": {"iou": 0.7, ...}, "per_chip": [...]}
    """
    with open(json_path) as f:
        data = json.load(f)

    # If the JSON has an "aggregate" key, use that (evaluate.py format)
    if isinstance(data, dict) and "aggregate" in data:
        data = data["aggregate"]

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="results/logs",
                        help="Directory containing *_results.json files")
    parser.add_argument("--output_csv", default="results/logs/all_results.csv")
    parser.add_argument("--output_md",  default="results/logs/all_results.md")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    rows = []

    # Walk every JSON file in the logs directory
    for jf in sorted(logs_dir.glob("*_results.json")):
        # Skip aggregated output files themselves (in case we run this twice)
        if jf.name.startswith("all_"):
            continue

        model, split = parse_filename(jf.stem)

        try:
            m = load_metrics(jf)
        except Exception as e:
            print(f"  [skip] {jf.name}: {e}")
            continue

        # Not all JSONs have every metric — use .get() with None defaults
        row = {
            "model":     model,
            "split":     split,
            "iou":       m.get("iou"),
            "dice":      m.get("dice"),
            "precision": m.get("precision"),
            "recall":    m.get("recall"),
            "f1":        m.get("f1"),
            "accuracy":  m.get("accuracy"),
            "tp":        m.get("tp"),
            "fp":        m.get("fp"),
            "tn":        m.get("tn"),
            "fn":        m.get("fn"),
            "modalities": m.get("modalities"),  # only present for ablations
            "source_file": jf.name,
        }
        rows.append(row)

    if not rows:
        print(f"No results found in {logs_dir}")
        return

    df = pd.DataFrame(rows)

    # Order: Otsu / FCN / Fusion / TriModal first, then ablations
    def sort_key(row):
        m = str(row["model"])
        if "otsu"       in m: return (0, m)
        if "fcn"        in m: return (1, m)
        if "fusion"     in m: return (2, m)
        if "trimodal"   in m: return (3, m)
        if "ablation"   in m: return (4, m)
        return (5, m)

    df["_sort"] = df.apply(sort_key, axis=1)
    df = df.sort_values(["_sort", "split"]).drop(columns="_sort")

    # Save CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows -> {args.output_csv}")

    # Build Markdown table (test split only, the publishable numbers)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) > 0:
        with open(args.output_md, "w") as f:
            f.write("# Test Set Results\n\n")
            f.write("| Model | IoU | Dice | Precision | Recall | F1 |\n")
            f.write("|-------|-----|------|-----------|--------|----|\n")
            for _, r in test_df.iterrows():
                name = r["model"]
                if r["modalities"]:
                    name = f"{name} [{r['modalities']}]"
                f.write(f"| {name} | "
                        f"{r['iou']:.4f} | "
                        f"{r['dice']:.4f} | "
                        f"{r['precision']:.4f} | "
                        f"{r['recall']:.4f} | "
                        f"{r['f1']:.4f} |\n")

        # Also dump the Bolivia (cross-region) table if present
        bol_df = df[df["split"] == "bolivia"]
        if len(bol_df) > 0:
            with open(args.output_md, "a") as f:
                f.write("\n\n# Bolivia (Cross-Region) Results\n\n")
                f.write("| Model | IoU | Dice | Precision | Recall |\n")
                f.write("|-------|-----|------|-----------|--------|\n")
                for _, r in bol_df.iterrows():
                    name = r["model"]
                    if r["modalities"]:
                        name = f"{name} [{r['modalities']}]"
                    f.write(f"| {name} | "
                            f"{r['iou']:.4f} | "
                            f"{r['dice']:.4f} | "
                            f"{r['precision']:.4f} | "
                            f"{r['recall']:.4f} |\n")

        print(f"Saved Markdown table -> {args.output_md}")

    # Console preview
    print("\n=== Test results ===")
    print(test_df[["model", "modalities", "iou", "dice",
                   "precision", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
