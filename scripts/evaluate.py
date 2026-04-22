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
    parser.add_argument("--model", type=str, required=True,
                        choices=["otsu", "fcn", "fusion", "trimodal", "ablation"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (.pt file)")
    parser.add_argument("--modalities", type=str, default=None,
                        help="Modalities string for --model=ablation (e.g. s1_s2_dem)")
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
    # weights_only=False required because new checkpoints include RNG state
    # (python random, numpy, torch) for exact training resume.
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
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


def evaluate_fusion(data_root, splits_dir, split, checkpoint, device):
    from src.models.fusion_unet import FusionUNet
    from src.data.dataset import Sen1Floods11MultiModal

    data_root = Path(data_root)
    splits_dir = Path(splits_dir)

    split_map = {
        "train": "flood_train_data.csv",
        "val": "flood_valid_data.csv",
        "test": "flood_test_data.csv",
        "bolivia": "flood_bolivia_data.csv",
    }
    split_csv = splits_dir / split_map[split]

    model = FusionUNet(s1_channels=2, s2_channels=13, num_classes=2).to(device)
    # weights_only=False required because new checkpoints include RNG state
    # (python random, numpy, torch) for exact training resume.
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint} (epoch {ckpt.get('epoch', '?')}, val_iou={ckpt.get('val_iou', '?'):.4f})")

    dataset = Sen1Floods11MultiModal(
        split_csv=split_csv,
        s1_dir=data_root / "S1Hand",
        s2_dir=data_root / "S2Hand",
        label_dir=data_root / "LabelHand",
        crop_size=None,
        augment=False,
        normalize=True,
    )

    accumulator = MetricAccumulator()
    results = []

    print(f"Evaluating Fusion U-Net on {split} ({len(dataset)} samples)...")
    for idx in tqdm(range(len(dataset))):
        s1, s2, label = dataset[idx]
        s1 = s1.unsqueeze(0).to(device)
        s2 = s2.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(s1, s2)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        label_np = label.numpy()

        metrics = compute_metrics(pred, label_np, ignore_index=255)
        metrics["file"] = dataset.samples[idx][0]
        results.append(metrics)
        accumulator.update(pred, label_np, ignore_index=255)

    aggregate = accumulator.compute()
    return results, aggregate


def evaluate_trimodal(data_root, splits_dir, split, checkpoint, device):
    """Evaluate the tri-modal (S1+S2+DEM) fusion model."""
    from src.models.trimodal_unet import TriModalFusionUNet
    from src.data.dataset import Sen1Floods11TriModal

    data_root = Path(data_root); splits_dir = Path(splits_dir)
    split_map = {
        "train": "flood_train_data.csv", "val":   "flood_valid_data.csv",
        "test":  "flood_test_data.csv",  "bolivia": "flood_bolivia_data.csv",
    }
    split_csv = splits_dir / split_map[split]

    model = TriModalFusionUNet(s1_channels=2, s2_channels=13, dem_channels=2,
                               num_classes=2).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {checkpoint} (epoch {ckpt.get('epoch','?')})")

    dataset = Sen1Floods11TriModal(
        split_csv=split_csv,
        s1_dir=data_root / "S1Hand",
        s2_dir=data_root / "S2Hand",
        dem_dir=data_root / "DEMHand",
        label_dir=data_root / "LabelHand",
        crop_size=None, augment=False, normalize=True,
    )

    acc, results = MetricAccumulator(), []
    print(f"Evaluating TriModal on {split} ({len(dataset)} samples)...")
    for idx in tqdm(range(len(dataset))):
        s1, s2, dem, label = dataset[idx]
        with torch.no_grad():
            out = model(s1.unsqueeze(0).to(device),
                        s2.unsqueeze(0).to(device),
                        dem.unsqueeze(0).to(device))
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
        m = compute_metrics(pred, label.numpy(), ignore_index=255)
        m["file"] = dataset.samples[idx][0]
        results.append(m)
        acc.update(pred, label.numpy(), ignore_index=255)
    return results, acc.compute()


def evaluate_ablation(data_root, splits_dir, split, checkpoint, modalities, device):
    """Evaluate a trained ablation variant (EarlyFusionUNet with N input channels).

    Uses the tri-modal dataset and concatenates only the requested modalities
    so the channel count matches what the checkpoint was trained on.
    """
    from src.models.early_fusion_unet import build_early_fusion
    from src.data.dataset import Sen1Floods11TriModal
    from scripts.train_ablation import select_modalities

    data_root = Path(data_root); splits_dir = Path(splits_dir)
    split_map = {
        "train": "flood_train_data.csv", "val":   "flood_valid_data.csv",
        "test":  "flood_test_data.csv",  "bolivia": "flood_bolivia_data.csv",
    }
    split_csv = splits_dir / split_map[split]

    model, in_ch = build_early_fusion(modalities, num_classes=2)
    model = model.to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {checkpoint} (modalities={modalities}, {in_ch}ch)")

    dataset = Sen1Floods11TriModal(
        split_csv=split_csv,
        s1_dir=data_root / "S1Hand",
        s2_dir=data_root / "S2Hand",
        dem_dir=data_root / "DEMHand",
        label_dir=data_root / "LabelHand",
        crop_size=None, augment=False, normalize=True,
    )

    acc, results = MetricAccumulator(), []
    print(f"Evaluating ablation [{modalities}] on {split} ({len(dataset)} samples)...")
    for idx in tqdm(range(len(dataset))):
        s1, s2, dem, label = dataset[idx]
        x = select_modalities(s1.unsqueeze(0), s2.unsqueeze(0),
                              dem.unsqueeze(0), modalities).to(device)
        with torch.no_grad():
            out = model(x)
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
        m = compute_metrics(pred, label.numpy(), ignore_index=255)
        m["file"] = dataset.samples[idx][0]
        results.append(m)
        acc.update(pred, label.numpy(), ignore_index=255)
    return results, acc.compute()


def main():
    args = parse_args()

    if args.model == "otsu":
        results, aggregate = evaluate_otsu(args.data_root, args.splits_dir, args.split)
    elif args.model in ("fcn", "fusion", "trimodal", "ablation"):
        if args.checkpoint is None:
            print(f"ERROR: --checkpoint required for {args.model} model")
            sys.exit(1)
        device_str = args.device
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)
        if args.model == "fcn":
            results, aggregate = evaluate_fcn(
                args.data_root, args.splits_dir, args.split, args.checkpoint, device,
            )
        elif args.model == "fusion":
            results, aggregate = evaluate_fusion(
                args.data_root, args.splits_dir, args.split, args.checkpoint, device,
            )
        elif args.model == "trimodal":
            results, aggregate = evaluate_trimodal(
                args.data_root, args.splits_dir, args.split, args.checkpoint, device,
            )
        elif args.model == "ablation":
            if args.modalities is None:
                print("ERROR: --modalities required for --model=ablation")
                sys.exit(1)
            results, aggregate = evaluate_ablation(
                args.data_root, args.splits_dir, args.split, args.checkpoint,
                args.modalities, device,
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
