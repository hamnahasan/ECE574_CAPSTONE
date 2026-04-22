"""Train Tri-Modal Fusion U-Net on Sen1Floods11 (S1 + S2 + DEM).

Phase 3: adds Copernicus DEM (elevation + slope) as a third encoder branch.
Cross-attention extended to 3-way fusion at each of the 4 encoder scales.

Usage:
    python scripts/train_trimodal.py \
        --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
        --output_dir results
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

from src.data.dataset import get_trimodal_dataloaders
from src.models.trimodal_unet import TriModalFusionUNet, count_parameters
from src.utils.metrics import MetricAccumulator
from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint, resolve_resume_path, save_history,
)


def parse_args():
    p = argparse.ArgumentParser()
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
    # Resume and checkpoint flags (for Isaac 24hr wall-time jobs).
    p.add_argument("--resume",      default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--auto_resume", action="store_true",
                   help="Auto-resume from latest checkpoint if exists")
    p.add_argument("--save_every",  type=int, default=5,
                   help="Save named checkpoint every N epochs")
    p.add_argument("--run_name",    default="trimodal_unet",
                   help="Run name prefix for checkpoints and logs")
    # Modality dropout probability (applied independently per modality per batch).
    # Scientific rationale: at inference time DEM or S2 may be unavailable
    # (cloud cover, missing tiles). Training with random modality dropout forces
    # the model to make predictions from incomplete inputs, improving robustness
    # and preventing over-reliance on any single modality.
    # Recommended: 0.1-0.2 (zero out one modality ~10-20% of batches).
    p.add_argument("--mod_dropout", type=float, default=0.1,
                   help="Probability of zeroing each modality during training")
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def apply_modality_dropout(s1, s2, dem, p):
    """Randomly zero out entire modality tensors during training.

    Scientific rationale:
    - Floods are often observed under cloud cover → S2 may be fully saturated
    - DEM tiles may be missing in some regions (ocean, data gaps)
    - SAR (S1) can have acquisition gaps or strong noise in mountainous terrain

    By randomly dropping each modality with probability p, the model is forced
    to learn complementary representations and cannot blindly rely on any one
    source. This also acts as a form of data augmentation.

    At inference, all modalities are always present (p=0), so the model uses
    the full signal when available but degrades gracefully when it is not.

    Args:
        s1, s2, dem: Input tensors (B, C, H, W).
        p: Probability of zeroing each modality independently.

    Returns:
        Augmented (s1, s2, dem) — some may be all-zeros.
    """
    import random
    # Each modality is dropped independently — on average p*3 modalities
    # are zeroed per batch, but the model always sees at least one.
    # We use torch.zeros_like to preserve device/dtype.
    if random.random() < p:
        s1  = torch.zeros_like(s1)
    if random.random() < p:
        s2  = torch.zeros_like(s2)
    if random.random() < p:
        dem = torch.zeros_like(dem)
    return s1, s2, dem


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, scaler, use_amp, mod_dropout_p=0.1):
    """Train one epoch and return (avg_loss, train_metrics).

    Train metrics are accumulated from training-set predictions and serve
    as the standard overfitting signal (compare against val_iou per epoch).
    """
    model.train()
    total_loss, n = 0.0, 0
    train_acc = MetricAccumulator()

    for s1, s2, dem, labels in tqdm(loader, desc="  Train", leave=False):
        s1 = s1.to(device); s2 = s2.to(device)
        dem = dem.to(device); labels = labels.to(device)

        # Apply modality dropout: randomly zero modalities to improve robustness.
        # Only applied during training, never during validation or testing.
        if mod_dropout_p > 0:
            s1, s2, dem = apply_modality_dropout(s1, s2, dem, mod_dropout_p)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(s1, s2, dem)
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

        # Free overfitting signal — predictions already computed
        with torch.no_grad():
            preds = torch.argmax(out, dim=1)
            train_acc.update(preds, labels, ignore_index=255)

    return total_loss / max(n, 1), train_acc.compute()


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, n = 0.0, 0
    acc = MetricAccumulator()
    for s1, s2, dem, labels in tqdm(loader, desc="  Val", leave=False):
        s1 = s1.to(device); s2 = s2.to(device)
        dem = dem.to(device); labels = labels.to(device)
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(s1, s2, dem)
            loss = criterion(out, labels)
        if not torch.isnan(loss):
            total_loss += loss.item(); n += 1
        acc.update(torch.argmax(out, dim=1), labels, ignore_index=255)
    return total_loss / max(n, 1), acc.compute()


def main():
    args   = parse_args()
    use_amp = not args.no_amp
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}  |  AMP: {use_amp}")

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    log_dir  = Path(args.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tri-modal data (S1 + S2 + DEM)...")
    loaders = get_trimodal_dataloaders(
        args.data_root, args.splits_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        crop_size=args.crop_size,
    )
    print(f"  Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    model = TriModalFusionUNet(s1_channels=2, s2_channels=13, dem_channels=2,
                               num_classes=2, attention_heads=4,
                               dropout_rate=0.1).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    class_weights = torch.tensor([1.0, 8.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=len(loaders["train"]) * args.epochs,
                                  eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    # Resume if checkpoint exists (Isaac 24hr jobs may require requeue).
    start_epoch, best_iou, history = 1, 0.0, []
    resume_path = resolve_resume_path(args.resume, args.auto_resume,
                                      ckpt_dir, args.run_name)
    if resume_path is not None:
        start_epoch, best_iou, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device,
        )

    if start_epoch > args.epochs:
        print(f"Already completed {args.epochs} epochs — skipping training.")
    else:
        print(f"\nTraining epochs {start_epoch}..{args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], criterion,
            optimizer, scheduler, device, scaler,
            use_amp, mod_dropout_p=args.mod_dropout,
        )
        val_loss, val_m = validate(model, loaders["val"], criterion, device, use_amp)
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

        # Rolling "latest" — used for auto-resume on requeue.
        save_checkpoint(
            ckpt_dir / f"{args.run_name}_latest.pt",
            epoch, model, optimizer, scheduler, scaler, best_iou, history,
            extra={"val_iou": val_m["iou"], "val_dice": val_m["dice"]},
        )
        save_history(history, log_dir / f"{args.run_name}_history.json")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_checkpoint(
                ckpt_dir / f"{args.run_name}_best.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"], "val_dice": val_m["dice"]},
            )
            print(f"  -> Saved best model (IoU: {val_m['iou']:.4f})")

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"{args.run_name}_epoch{epoch}.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"]},
            )

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")

    # Test evaluation
    print("\nEvaluating best model on test set...")
    ckpt = torch.load(ckpt_dir / f"{args.run_name}_best.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_m = validate(model, loaders["test"], criterion, device, use_amp)

    print(f"Test IoU:  {test_m['iou']:.4f}")
    print(f"Test Dice: {test_m['dice']:.4f}")
    print(f"Test Prec: {test_m['precision']:.4f}")
    print(f"Test Rec:  {test_m['recall']:.4f}")
    print(f"Test Acc:  {test_m['accuracy']:.4f}")

    with open(log_dir / f"{args.run_name}_test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)


if __name__ == "__main__":
    main()
