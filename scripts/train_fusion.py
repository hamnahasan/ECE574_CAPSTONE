"""Train Dual-Encoder Fusion U-Net on Sen1Floods11 (S1 + S2).

Cross-attention fusion model with separate S1/S2 encoder branches.
Builds on the baseline training setup with adjustments for multi-modal input:
- Lower batch size (4 vs 8) to fit dual encoders in 8GB VRAM
- Mixed precision (AMP) to reduce memory footprint
- Same optimizer/scheduler/loss as baseline for fair comparison

Usage:
    python scripts/train_fusion.py \
        --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
        --output_dir results
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
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_multimodal_dataloaders
from src.models.fusion_unet import FusionUNet, count_parameters
from src.utils.metrics import MetricAccumulator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fusion U-Net (S1+S2)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to HandLabeled directory")
    parser.add_argument("--splits_dir", type=str, required=True,
                        help="Path to splits/flood_handlabeled directory")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision training (default: on)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision")
    return parser.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, use_amp):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for s1, s2, labels in pbar:
        s1 = s1.to(device)
        s2 = s2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(s1, s2)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        loss_val = loss.item()
        if not (loss_val != loss_val):  # NaN check
            total_loss += loss_val
            num_batches += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.6f}",
        )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0
    num_batches = 0
    metrics = MetricAccumulator()

    for s1, s2, labels in tqdm(loader, desc="  Val", leave=False):
        s1 = s1.to(device)
        s2 = s2.to(device)
        labels = labels.to(device)

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(s1, s2)
            loss = criterion(outputs, labels)

        if not torch.isnan(loss):
            total_loss += loss.item()
            num_batches += 1

        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, labels, ignore_index=255)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, metrics.compute()


def main():
    args = parse_args()
    use_amp = args.amp and not args.no_amp
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    # Paths
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading multi-modal data (S1 + S2)...")
    loaders = get_multimodal_dataloaders(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
    )
    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Val:   {len(loaders['val'].dataset)} samples")
    print(f"  Test:  {len(loaders['test'].dataset)} samples")

    # Model
    model = FusionUNet(
        s1_channels=2,
        s2_channels=13,
        num_classes=2,
        attention_heads=4,
        dropout_rate=0.1,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss (same weighting as baseline for fair comparison)
    class_weights = torch.tensor([1.0, 8.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    # Optimizer + scheduler (cosine decay, no restarts — attention is unstable with LR spikes)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = len(loaders["train"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=steps_per_epoch * args.epochs,
        eta_min=1e-6,
    )

    scaler = GradScaler(enabled=use_amp)

    # Training loop
    best_iou = 0.0
    history = []
    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, scaler, use_amp,
        )
        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device, use_amp,
        )

        elapsed = time.time() - t0
        iou = val_metrics["iou"]
        dice = val_metrics["dice"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {iou:.4f} | "
            f"Val Dice: {dice:.4f} | "
            f"{elapsed:.1f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
        }
        history.append(record)

        # Save best model
        if iou > best_iou:
            best_iou = iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": iou,
                "val_dice": dice,
            }, ckpt_dir / "fusion_unet_best.pt")
            print(f"  → Saved best model (IoU: {iou:.4f})")

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_iou": iou,
            }, ckpt_dir / f"fusion_unet_epoch{epoch}.pt")

    # Save final + history
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
    }, ckpt_dir / "fusion_unet_final.pt")

    with open(log_dir / "fusion_unet_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs: {log_dir / 'fusion_unet_history.json'}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    ckpt = torch.load(ckpt_dir / "fusion_unet_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_metrics = validate(model, loaders["test"], criterion, device, use_amp)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU:  {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test Prec: {test_metrics['precision']:.4f}")
    print(f"Test Rec:  {test_metrics['recall']:.4f}")
    print(f"Test Acc:  {test_metrics['accuracy']:.4f}")

    with open(log_dir / "fusion_unet_test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
