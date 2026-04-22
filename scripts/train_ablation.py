"""Ablation study training script for modality contribution analysis.

SCIENTIFIC PURPOSE:
    The ablation study answers the core question reviewers will ask:
    "Does each modality actually contribute to performance?"

    We train the same architecture with every subset of modalities:
        - S1 only       (2ch)  — SAR alone
        - S2 only       (13ch) — optical alone
        - DEM only      (2ch)  — topography alone
        - S1 + S2       (15ch) — SAR + optical (Phase 2 comparison)
        - S1 + DEM      (4ch)  — SAR + topography
        - S2 + DEM      (15ch) — optical + topography
        - S1 + S2 + DEM (17ch) — full tri-modal (Phase 3)

    For 2-modality combinations we use the EarlyFusionUNet (concat input)
    rather than the dual-encoder cross-attention model, so that the ONLY
    variable between ablation runs is which modalities are present — not
    the architecture. This gives a clean, fair comparison.

    Expected findings:
    - S1+S2+DEM > S1+S2 > S1 only  (each modality adds value)
    - DEM alone will be poor (no texture/radiometry)
    - S1+DEM may outperform S1 in cloudy/ambiguous scenes
    - S2+DEM may be strongest in clear-sky conditions

Usage:
    # Train one ablation variant
    python scripts/train_ablation.py \
        --modalities s1_s2_dem \
        --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled

    # Available modalities strings:
    #   s1 | s2 | dem | s1_s2 | s1_dem | s2_dem | s1_s2_dem
"""

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.early_fusion_unet import build_early_fusion, count_parameters
from src.utils.metrics import MetricAccumulator
from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint, resolve_resume_path, save_history,
)

# Valid modality combinations and their total channel counts.
# Kept here as a reference for the paper's Table of ablation variants.
MODALITY_CHANNELS = {
    "s1":        2,   # VV + VH SAR backscatter
    "s2":        13,  # 13 multispectral bands
    "dem":       2,   # elevation (m) + slope (degrees)
    "s1_s2":     15,  # SAR + optical — direct comparison to Phase 2
    "s1_dem":    4,   # SAR + topography — tests DEM without optical
    "s2_dem":    15,  # optical + topography — tests if SAR is necessary
    "s1_s2_dem": 17,  # full input — expected best performance
}


def parse_args():
    p = argparse.ArgumentParser(description="Ablation study: train with modality subset")
    p.add_argument("--modalities",  required=True, choices=list(MODALITY_CHANNELS.keys()),
                   help="Which modalities to use (e.g. s1_s2_dem)")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--crop_size",   type=int,   default=256)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--device",      default="auto")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--resume",      default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--auto_resume", action="store_true",
                   help="Auto-resume from latest checkpoint if exists")
    p.add_argument("--save_every",  type=int, default=5,
                   help="Save named checkpoint every N epochs")
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_dataloaders(modalities, data_root, splits_dir, batch_size, num_workers, crop_size):
    """Build dataloaders that return only the requested modalities.

    We always load all three modalities from disk and select the needed ones
    at collation time. This avoids maintaining separate dataset classes per
    ablation variant and ensures the train/val/test splits are identical
    across all ablation runs (same random seeds, same chip ordering).
    """
    from src.data.dataset import get_trimodal_dataloaders

    # Load full tri-modal data regardless of which modalities are requested.
    # The collate_fn below will select and concatenate only the needed ones.
    loaders = get_trimodal_dataloaders(
        data_root, splits_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        crop_size=crop_size,
    )
    return loaders


def select_modalities(s1, s2, dem, modalities):
    """Concatenate only the requested modality tensors.

    This function is the core of the ablation: by controlling which tensors
    are concatenated, we can test any combination without changing the model
    architecture. The EarlyFusionUNet simply receives a different channel count.

    Args:
        s1:  (B, 2,  H, W) — SAR VV/VH
        s2:  (B, 13, H, W) — optical bands
        dem: (B, 2,  H, W) — elevation + slope
        modalities: str like 's1_s2_dem'

    Returns:
        (B, C, H, W) concatenated tensor where C = sum of selected channels.
    """
    parts = []
    mods = modalities.split("_")
    if "s1"  in mods: parts.append(s1)
    if "s2"  in mods: parts.append(s2)
    if "dem" in mods: parts.append(dem)
    return torch.cat(parts, dim=1)


def train_one_epoch(model, loader, modalities, criterion, optimizer,
                    scheduler, device, scaler, use_amp):
    """Train one epoch and return (avg_loss, train_metrics).

    Train metrics serve as the overfitting signal across the 7 ablation
    variants — a model receiving more modalities should reach higher
    train_iou faster but should NOT show a larger train-vs-val gap.
    """
    model.train()
    total_loss, n = 0.0, 0
    train_acc = MetricAccumulator()

    for s1, s2, dem, labels in tqdm(loader, desc="  Train", leave=False):
        s1 = s1.to(device); s2 = s2.to(device)
        dem = dem.to(device); labels = labels.to(device)

        # Build input tensor for this ablation variant
        x = select_modalities(s1, s2, dem, modalities)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(x)
            loss = criterion(out, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        lv = loss.item()
        if lv == lv:  # NaN guard
            total_loss += lv; n += 1

        with torch.no_grad():
            preds = torch.argmax(out, dim=1)
            train_acc.update(preds, labels, ignore_index=255)

    return total_loss / max(n, 1), train_acc.compute()


@torch.no_grad()
def validate(model, loader, modalities, criterion, device, use_amp):
    model.eval()
    total_loss, n = 0.0, 0
    acc = MetricAccumulator()

    for s1, s2, dem, labels in tqdm(loader, desc="  Val", leave=False):
        s1 = s1.to(device); s2 = s2.to(device)
        dem = dem.to(device); labels = labels.to(device)
        x = select_modalities(s1, s2, dem, modalities)

        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(x)
            loss = criterion(out, labels)

        if not torch.isnan(loss):
            total_loss += loss.item(); n += 1
        acc.update(torch.argmax(out, dim=1), labels, ignore_index=255)

    return total_loss / max(n, 1), acc.compute()


def main():
    args    = parse_args()
    use_amp = not args.no_amp
    set_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
              if args.device == "auto" else torch.device(args.device)

    print(f"Ablation: modalities={args.modalities}  "
          f"channels={MODALITY_CHANNELS[args.modalities]}")
    print(f"Device: {device}  |  AMP: {use_amp}")

    # Output paths: separate subdirectory per ablation variant so all
    # runs can be submitted in parallel on Isaac without collision.
    run_name = f"ablation_{args.modalities}"
    ckpt_dir = Path(args.output_dir) / "checkpoints" / run_name
    log_dir  = Path(args.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data...")
    loaders = get_dataloaders(args.modalities, args.data_root, args.splits_dir,
                              args.batch_size, args.num_workers, args.crop_size)
    print(f"  Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # Build EarlyFusionUNet with the correct input channel count.
    # All ablation variants use the same architecture; only in_channels differs.
    # This ensures any IoU difference is attributable to the modalities alone.
    model, in_ch = build_early_fusion(args.modalities, num_classes=2)
    model = model.to(device)
    print(f"Model: EarlyFusionUNet({in_ch}ch) — {count_parameters(model):,} parameters")

    # Same loss, optimizer, and scheduler as Phase 2 and 3 for fair comparison.
    class_weights = torch.tensor([1.0, 8.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=len(loaders["train"]) * args.epochs,
                                  eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    # Resume if checkpoint exists.
    start_epoch, best_iou, history = 1, 0.0, []
    resume_path = resolve_resume_path(args.resume, args.auto_resume,
                                      ckpt_dir, run_name)
    if resume_path is not None:
        start_epoch, best_iou, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device,
        )

    if start_epoch > args.epochs:
        print(f"Already completed {args.epochs} epochs — skipping training.")
    else:
        print(f"\nTraining {run_name} epochs {start_epoch}..{args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], args.modalities,
            criterion, optimizer, scheduler,
            device, scaler, use_amp,
        )
        val_loss, val_m = validate(model, loaders["val"], args.modalities,
                                   criterion, device, use_amp)
        elapsed = time.time() - t0
        iou_gap = train_m["iou"] - val_m["iou"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} (IoU {train_m['iou']:.3f}) | "
              f"Val: {val_loss:.4f} (IoU {val_m['iou']:.3f}, Dice {val_m['dice']:.3f}) | "
              f"Gap: {iou_gap:+.3f} | {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_iou":  train_m["iou"],
                        "train_dice": train_m["dice"],
                        **{f"val_{k}": v for k, v in val_m.items()},
                        "iou_gap": iou_gap,
                        "lr": scheduler.get_last_lr()[0]})

        # Rolling latest for auto-resume.
        save_checkpoint(
            ckpt_dir / f"{run_name}_latest.pt",
            epoch, model, optimizer, scheduler, scaler, best_iou, history,
            extra={"val_iou": val_m["iou"], "val_dice": val_m["dice"],
                   "modalities": args.modalities},
        )
        save_history(history, log_dir / f"{run_name}_history.json")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_checkpoint(
                ckpt_dir / f"{run_name}_best.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"], "val_dice": val_m["dice"],
                       "modalities": args.modalities},
            )
            print(f"  -> Saved best (IoU: {val_m['iou']:.4f})")

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"{run_name}_epoch{epoch}.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"],
                       "modalities": args.modalities},
            )

    print(f"\nDone. Best Val IoU: {best_iou:.4f}")

    # Test evaluation
    print("\nEvaluating best model on test set...")
    ckpt = torch.load(ckpt_dir / f"{run_name}_best.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_m = validate(model, loaders["test"], args.modalities,
                         criterion, device, use_amp)

    # Save test results with modality label so they can be compiled into
    # the ablation table in the paper automatically.
    test_m["modalities"] = args.modalities
    test_m["in_channels"] = in_ch
    test_m["best_val_iou"] = best_iou

    with open(log_dir / f"{run_name}_test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)

    print(f"[{args.modalities}] Test IoU: {test_m['iou']:.4f}  "
          f"Dice: {test_m['dice']:.4f}  "
          f"Prec: {test_m['precision']:.4f}  "
          f"Rec: {test_m['recall']:.4f}")


if __name__ == "__main__":
    main()
