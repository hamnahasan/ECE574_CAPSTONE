"""Evaluate a trained model or Otsu baseline on Sen1Floods11 splits.

Usage:
    # Evaluate Otsu baseline
    python scripts/evaluate.py \
        --model otsu \
        --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
        --split test

    # Evaluate FCN baseline
    python scripts/evaluate.py \
        --model fcn \
        --checkpoint results/checkpoints/fcn_baseline_best.pt \
        --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
        --split test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import compute_metrics, MetricAccumulator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate flood segmentation model")
    parser.add_argument("--model", type=str, required=True, choices=["otsu", "fcn"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (for fcn)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test", "bolivia"],
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def evaluate_otsu(data_root, splits_dir, split):
    from src.models.otsu_baseline import evaluate_otsu_on_split

    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    s1_dir = data_root / "S1Hand"
    label_dir = data_root / "LabelHand"

    split_map = {
        "train": "flood_train_data.csv",
        "val": "flood_valid_data.csv",
        "test": "flood_test_data.csv",
        "bolivia": "flood_bolivia_data.csv",
    }
    split_csv = splits_dir / split_map[split]

    print(f"Evaluating Otsu on {split} ({split_csv})...")
    results, aggregate = evaluate_otsu_on_split(
        s1_dir=s1_dir,
        label_dir=label_dir,
        split_csv=split_csv,
        band=1,  # VH
    )
    return results, aggregate


def evaluate_fcn(data_root, splits_dir, split, checkpoint, device):
    import pandas as pd
    import rasterio
    from src.models.fcn_baseline import FCNBaseline
    from src.data.dataset import Sen1Floods11

    data_root = Path(data_root)
    splits_dir = Path(splits_dir)

    split_map = {
        "train": "flood_train_data.csv",
        "val": "flood_valid_data.csv",
        "test": "flood_test_data.csv",
        "bolivia": "flood_bolivia_data.csv",
    }
    split_csv = splits_dir / split_map[split]

    # Load model
    model = FCNBaseline(in_channels=2, num_classes=2).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Dataset (no crop, no augment)
    dataset = Sen1Floods11(
        split_csv=split_csv,
        s1_dir=data_root / "S1Hand",
        label_dir=data_root / "LabelHand",
        crop_size=None,
        augment=False,
        normalize=True,
    )

    accumulator = MetricAccumulator()
    results = []

    print(f"Evaluating FCN on {split} ({len(dataset)} samples)...")
    for idx in tqdm(range(len(dataset))):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        label_np = label.numpy()

        metrics = compute_metrics(pred, label_np, ignore_index=255)
        metrics["file"] = dataset.samples[idx][0]
        results.append(metrics)
        accumulator.update(pred, label_np, ignore_index=255)

    aggregate = accumulator.compute()
    return results, aggregate


def main():
    args = parse_args()

    if args.model == "otsu":
        results, aggregate = evaluate_otsu(args.data_root, args.splits_dir, args.split)
    elif args.model == "fcn":
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for fcn model")
            sys.exit(1)
        device_str = args.device
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)
        results, aggregate = evaluate_fcn(
            args.data_root, args.splits_dir, args.split, args.checkpoint, device,
        )

    # Print results
    print(f"\n{'='*50}")
    print(f"  {args.model.upper()} — {args.split} set results")
    print(f"{'='*50}")
    print(f"  IoU:       {aggregate['iou']:.4f}")
    print(f"  Dice:      {aggregate['dice']:.4f}")
    print(f"  Precision: {aggregate['precision']:.4f}")
    print(f"  Recall:    {aggregate['recall']:.4f}")
    print(f"  F1:        {aggregate['f1']:.4f}")
    print(f"  Accuracy:  {aggregate['accuracy']:.4f}")
    print(f"  TP: {aggregate['tp']:,}  FP: {aggregate['fp']:,}  "
          f"TN: {aggregate['tn']:,}  FN: {aggregate['fn']:,}")

    # Save
    output_path = args.output
    if output_path is None:
        output_path = f"results/logs/{args.model}_{args.split}_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "aggregate": aggregate,
            "per_chip": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
