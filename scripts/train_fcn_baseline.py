"""Train FCN-ResNet50 baseline on Sen1Floods11 (S1-only).

Replicates the deep-learning baseline from Bonafilia et al. (2020):
- FCN-ResNet50 with first conv adapted to 2 channels (VV/VH)
- BatchNorm replaced with GroupNorm (stable for small batch sizes)
- Weighted CrossEntropyLoss (1:8 for non-water:water)
- AdamW + CosineAnnealingLR scheduler
- 100 epochs, 256x256 random crops, random H/V flips

WHY CosineAnnealingLR (not WarmRestarts as in original)?
    The original paper used CosineAnnealingWarmRestarts, which periodically
    spikes the LR back to its initial value. We found this spike destabilizes
    attention-based models in Phase 2/3. For a fair architecture-controlled
    comparison across all our experiments, we use the same scheduler
    (CosineAnnealingLR) for FCN, Fusion, and TriModal. The FCN baseline
    achieves the same Test IoU (~0.638) under either scheduler, so this
    change does not advantage or disadvantage the baseline.

WHY track train IoU as well as val IoU?
    The gap between train and val IoU is the canonical overfitting signal.
    A model with high train_iou but low val_iou is overfitting; a model
    with both low is underfitting. We accumulate metrics during training
    (free — predictions are computed anyway) so the analysis notebook can
    plot train_iou vs val_iou per epoch and flag overfitting visually.

Usage:
    python scripts/train_fcn_baseline.py \
        --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
        --output_dir results --auto_resume
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataloaders
from src.models.fcn_baseline import FCNBaseline
from src.utils.metrics import MetricAccumulator
from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint, resolve_resume_path, save_history,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FCN baseline (S1-only)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Initial LR (was 5e-4 in original; lowered for "
                             "consistency with attention-based models)")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    # Isaac/SLURM resume infrastructure (matches train_fusion / train_trimodal)
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Auto-resume from latest checkpoint if exists")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save named checkpoint every N epochs")
    parser.add_argument("--run_name", type=str, default="fcn_baseline")
    return parser.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Train for one epoch and return (avg_loss, train_metrics).

    Train metrics are accumulated from the model's predictions during
    training. They are slightly biased (augmented data, dropout active)
    but provide the standard "train accuracy" signal used to detect
    overfitting against the val metrics.
    """
    model.train()
    total_loss, n = 0.0, 0
    train_acc = MetricAccumulator()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device); labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        if lv == lv:  # NaN guard
            total_loss += lv; n += 1

        # Free overfitting signal: accumulate train predictions
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            train_acc.update(preds, labels, ignore_index=255)

        pbar.set_postfix(loss=f"{lv:.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    return total_loss / max(n, 1), train_acc.compute()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    metrics = MetricAccumulator()

    for images, labels in tqdm(loader, desc="  Val", leave=False):
        images = images.to(device); labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if not torch.isnan(loss):
            total_loss += loss.item(); n += 1
        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, labels, ignore_index=255)

    return total_loss / max(n, 1), metrics.compute()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir  = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    loaders = get_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        crop_size=args.crop_size,
    )
    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Val:   {len(loaders['val'].dataset)} samples")
    print(f"  Test:  {len(loaders['test'].dataset)} samples")

    model = FCNBaseline(in_channels=2, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    class_weights = torch.tensor([1.0, 8.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(loaders["train"]) * args.epochs,
        eta_min=1e-6,
    )

    # Resume if checkpoint exists
    start_epoch, best_iou, history = 1, 0.0, []
    resume_path = resolve_resume_path(
        args.resume, args.auto_resume, ckpt_dir, args.run_name,
    )
    if resume_path is not None:
        # FCN doesn't use AMP, so scaler is None
        start_epoch, best_iou, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, None, device,
        )

    if start_epoch > args.epochs:
        print(f"Already completed {args.epochs} epochs — skipping training.")
    else:
        print(f"\nTraining epochs {start_epoch}..{args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, device,
        )
        val_loss, val_m = validate(model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        # Overfitting gap: train_iou - val_iou.
        # Positive and growing = overfitting; near zero or growing val_iou = healthy.
        iou_gap = train_m["iou"] - val_m["iou"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} (IoU {train_m['iou']:.3f}) | "
            f"Val: {val_loss:.4f} (IoU {val_m['iou']:.3f}, Dice {val_m['dice']:.3f}) | "
            f"Gap: {iou_gap:+.3f} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_iou":  train_m["iou"],
            "train_dice": train_m["dice"],
            **{f"val_{k}": v for k, v in val_m.items()},
            "iou_gap": iou_gap,
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
        })

        # Rolling latest for auto-resume
        save_checkpoint(
            ckpt_dir / f"{args.run_name}_latest.pt",
            epoch, model, optimizer, scheduler, None, best_iou, history,
            extra={"val_iou": val_m["iou"], "val_dice": val_m["dice"]},
        )
        save_history(history, log_dir / f"{args.run_name}_history.json")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_checkpoint(
                ckpt_dir / f"{args.run_name}_best.pt",
                epoch, model, optimizer, scheduler, None, best_iou, history,
                extra={"val_iou": val_m["iou"], "val_dice": val_m["dice"]},
            )
            print(f"  -> Saved best (IoU: {val_m['iou']:.4f})")

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"{args.run_name}_epoch{epoch}.pt",
                epoch, model, optimizer, scheduler, None, best_iou, history,
                extra={"val_iou": val_m["iou"]},
            )

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")

    # Test evaluation
    print("\nEvaluating best model on test set...")
    ckpt = torch.load(ckpt_dir / f"{args.run_name}_best.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_m = validate(model, loaders["test"], criterion, device)
    print(f"Test IoU:  {test_m['iou']:.4f}")
    print(f"Test Dice: {test_m['dice']:.4f}")
    print(f"Test Prec: {test_m['precision']:.4f}")
    print(f"Test Rec:  {test_m['recall']:.4f}")

    with open(log_dir / f"{args.run_name}_test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)


if __name__ == "__main__":
    main()
