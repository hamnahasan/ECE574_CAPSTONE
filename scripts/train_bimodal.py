"""Train a BimodalCrossAttnUNet for any 2-modality pair from {s1, s2, dem}.

Closes the cross-attention ablation matrix (paper experiment A3). Existing
runs already cover s1+s2 (FusionUNet) and s1+s2+dem (TriModalFusionUNet);
this script adds the missing s1+dem and s2+dem cells.

Why a separate script and not extending train_fusion.py:
  - train_fusion.py is hardcoded to S1+S2 (uses get_multimodal_dataloaders
    which only knows about HandLabeled/S1Hand+S2Hand).
  - The bimodal cross-attention model is generic over which two modalities
    it takes, but the disk layout for DEM is different (DEMHand/) and the
    data class returns a flat (s1, s2, dem, label) tuple.
  - Cleanest path: load the trimodal dataloader, drop one modality at the
    Python level, hand the remaining two to BimodalCrossAttnUNet.

Usage:
    python scripts/train_bimodal.py --modalities s1_dem ...
    python scripts/train_bimodal.py --modalities s2_dem ...
    python scripts/train_bimodal.py --modalities s1_s2  ...   # for parity reruns
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
from src.models.bimodal_cross_attn_unet import build_bimodal, count_parameters
from src.utils.metrics import MetricAccumulator
from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint, resolve_resume_path, save_history,
)


# Map modality keys to their position in the (s1, s2, dem) tuple returned
# by the trimodal dataloader.
MODALITY_INDEX = {"s1": 0, "s2": 1, "dem": 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modalities", required=True,
                   choices=["s1_s2", "s1_dem", "s2_dem"],
                   help="Underscore-joined pair, in attention order (a_b)")
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
    p.add_argument("--resume",      default=None)
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--save_every",  type=int, default=5)
    # Modality-dropout probability — applies only to 2-modality models, where
    # zeroing one would leave the model with nothing. Default 0 (off). Kept
    # as a flag for symmetry with train_trimodal so the multi-seed harness
    # can pass through unchanged.
    p.add_argument("--mod_dropout", type=float, default=0.0,
                   help="Probability of zeroing the 'b' modality during training")
    p.add_argument("--run_name",    default=None,
                   help="Override; defaults to bimodal_<modalities>")
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def select_pair(s1, s2, dem, modalities):
    """Pick the two modalities the model expects, in declared order."""
    a_key, b_key = modalities.split("_")
    pool = {"s1": s1, "s2": s2, "dem": dem}
    return pool[a_key], pool[b_key]


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, scaler, use_amp, modalities, mod_dropout_p):
    model.train()
    total_loss, n = 0.0, 0
    acc = MetricAccumulator()
    pbar = tqdm(loader, desc="  Train", leave=False)
    for s1, s2, dem, labels in pbar:
        s1, s2, dem = s1.to(device), s2.to(device), dem.to(device)
        labels = labels.to(device)
        a, b = select_pair(s1, s2, dem, modalities)

        # Modality dropout — only zero the 'b' tensor, never both.
        if mod_dropout_p > 0:
            import random
            if random.random() < mod_dropout_p:
                b = torch.zeros_like(b)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(a, b)
            loss = criterion(out, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if not torch.isnan(loss):
            total_loss += loss.item(); n += 1
        with torch.no_grad():
            acc.update(torch.argmax(out, dim=1), labels, ignore_index=255)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(n, 1), acc.compute()


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, modalities):
    model.eval()
    total_loss, n = 0.0, 0
    acc = MetricAccumulator()
    for s1, s2, dem, labels in tqdm(loader, desc="  Val", leave=False):
        s1, s2, dem = s1.to(device), s2.to(device), dem.to(device)
        labels = labels.to(device)
        a, b = select_pair(s1, s2, dem, modalities)
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(a, b)
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
    print(f"Device: {device}  AMP: {use_amp}  Modalities: {args.modalities}")

    run_name = args.run_name or f"bimodal_{args.modalities}"
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    log_dir  = Path(args.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tri-modal data (will select pair internally)...")
    loaders = get_trimodal_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        crop_size=args.crop_size,
    )
    print(f"  Train: {len(loaders['train'].dataset)}  "
          f"Val: {len(loaders['val'].dataset)}  "
          f"Test: {len(loaders['test'].dataset)}")

    model = build_bimodal(tuple(args.modalities.split("_"))).to(device)
    print(f"Model: BimodalCrossAttnUNet({args.modalities}) — "
          f"{count_parameters(model):,} parameters")

    class_weights = torch.tensor([1.0, 8.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=len(loaders["train"]) * args.epochs,
                                  eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    start_epoch, best_iou, history = 1, 0.0, []
    resume_path = resolve_resume_path(args.resume, args.auto_resume,
                                      ckpt_dir, run_name)
    if resume_path is not None:
        start_epoch, best_iou, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device,
        )

    if start_epoch > args.epochs:
        print(f"Already completed {args.epochs} epochs — skipping.")
    else:
        print(f"\nTraining {run_name}: epochs {start_epoch}..{args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, scaler, use_amp, args.modalities, args.mod_dropout,
        )
        val_loss, val_m = validate(
            model, loaders["val"], criterion, device, use_amp, args.modalities,
        )
        elapsed = time.time() - t0
        iou_gap = train_m["iou"] - val_m["iou"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train {train_loss:.4f} (IoU {train_m['iou']:.3f}) | "
              f"Val {val_loss:.4f} (IoU {val_m['iou']:.3f}) | "
              f"Gap {iou_gap:+.3f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_iou":  train_m["iou"], "train_dice": train_m["dice"],
            **{f"val_{k}": v for k, v in val_m.items()},
            "iou_gap": iou_gap,
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
        })

        save_checkpoint(
            ckpt_dir / f"{run_name}_latest.pt",
            epoch, model, optimizer, scheduler, scaler, best_iou, history,
            extra={"val_iou": val_m["iou"]},
        )
        save_history(history, log_dir / f"{run_name}_history.json")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_checkpoint(
                ckpt_dir / f"{run_name}_best.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"]},
            )
            print(f"  -> Saved best model (IoU: {best_iou:.4f})")

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"{run_name}_epoch{epoch}.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"]},
            )

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")

    print("\nEvaluating best model on test set...")
    ckpt = torch.load(ckpt_dir / f"{run_name}_best.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_m = validate(model, loaders["test"], criterion, device, use_amp,
                         args.modalities)
    print(f"Test IoU:  {test_m['iou']:.4f}")
    print(f"Test Dice: {test_m['dice']:.4f}")
    print(f"Test Prec: {test_m['precision']:.4f}")
    print(f"Test Rec:  {test_m['recall']:.4f}")

    with open(log_dir / f"{run_name}_test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)


if __name__ == "__main__":
    main()
