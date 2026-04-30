"""Train classical segmentation baselines using segmentation_models_pytorch (smp).

Wraps SMP's library models (Unet, UnetPlusPlus, DeepLabV3Plus, FPN, etc.) so
they train on Sen1Floods11 with the same loaders, optimizer, and schedule as
our headline TriModal model. Apples-to-apples literature comparison.

Architectures covered:
    --arch unet            Vanilla U-Net with ResNet34 encoder (Ronneberger 2015)
    --arch unetplusplus    U-Net++ (Zhou 2018)
    --arch deeplabv3plus   DeepLabV3+ (Chen 2018)
    --arch fpn             Feature Pyramid Network
    --arch manet           MAnet (Fan 2020)

Modality input is concatenated at the input layer; first conv is replaced
to accept the right channel count. This matches how a non-fusion-aware
baseline would handle multi-modal input.

Requires: pip install segmentation-models-pytorch  (conda env: floodseg)

Usage:
    python scripts/literature_baselines/train_smp_baselines.py \\
        --arch unet --modalities s1_s2_dem \\
        --data_root  ... --splits_dir ...
"""

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("ERROR: segmentation_models_pytorch is required for this script.")
    print("Install it in the floodseg conda env on Isaac:")
    print("    pip install segmentation-models-pytorch")
    sys.exit(1)

from src.data.dataset import get_trimodal_dataloaders
from src.utils.metrics import MetricAccumulator
from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint, resolve_resume_path, save_history,
)


MODALITY_CHANNELS = {"s1": 2, "s2": 13, "dem": 2}

# Map our --arch flag to SMP's model classes.
SMP_BUILDERS = {
    "unet":           smp.Unet,
    "unetplusplus":   smp.UnetPlusPlus,
    "deeplabv3plus":  smp.DeepLabV3Plus,
    "fpn":            smp.FPN,
    "manet":          smp.MAnet,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",        required=True, choices=list(SMP_BUILDERS))
    p.add_argument("--modalities",  default="s1_s2",
                   help="Underscore-joined modality keys; concatenated at input")
    p.add_argument("--encoder",     default="resnet34",
                   help="SMP encoder name (resnet34, resnet50, etc.)")
    p.add_argument("--encoder_weights", default="imagenet",
                   help="Encoder pretraining; set to 'none' to disable")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--crop_size",   type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--save_every",  type=int, default=5)
    p.add_argument("--run_name",    default=None)
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_smp_model(arch, in_channels, encoder, encoder_weights):
    """Construct an SMP segmentation model with the given input channel count."""
    builder = SMP_BUILDERS[arch]
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights=None if encoder_weights == "none" else encoder_weights,
        in_channels=in_channels,
        classes=2,
    )
    return builder(**kwargs)


def select_and_concat(s1, s2, dem, modalities):
    pool = {"s1": s1, "s2": s2, "dem": dem}
    keys = modalities.split("_")
    return torch.cat([pool[k] for k in keys], dim=1)


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, scaler, use_amp, modalities):
    model.train()
    total_loss, n = 0.0, 0
    acc = MetricAccumulator()
    for s1, s2, dem, labels in tqdm(loader, desc="  Train", leave=False):
        s1, s2, dem = s1.to(device), s2.to(device), dem.to(device)
        labels = labels.to(device)
        x = select_and_concat(s1, s2, dem, modalities)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(x)
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
    return total_loss / max(n, 1), acc.compute()


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, modalities):
    model.eval()
    total_loss, n = 0.0, 0
    acc = MetricAccumulator()
    for s1, s2, dem, labels in tqdm(loader, desc="  Val", leave=False):
        s1, s2, dem = s1.to(device), s2.to(device), dem.to(device)
        labels = labels.to(device)
        x = select_and_concat(s1, s2, dem, modalities)
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(x)
            loss = criterion(out, labels)
        if not torch.isnan(loss):
            total_loss += loss.item(); n += 1
        acc.update(torch.argmax(out, dim=1), labels, ignore_index=255)
    return total_loss / max(n, 1), acc.compute()


def main():
    args = parse_args()
    use_amp = not args.no_amp
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = sum(MODALITY_CHANNELS[m] for m in args.modalities.split("_"))
    run_name = args.run_name or f"smp_{args.arch}_{args.encoder}_{args.modalities}"
    print(f"Building {run_name}: {args.arch}/{args.encoder} "
          f"in_ch={in_channels} weights={args.encoder_weights}")

    model = build_smp_model(args.arch, in_channels,
                            args.encoder, args.encoder_weights).to(device)

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    log_dir  = Path(args.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_trimodal_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        crop_size=args.crop_size,
    )

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.0], device=device),
                                     ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=len(loaders["train"]) * args.epochs,
                                  eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    start_epoch, best_iou, history = 1, 0.0, []
    resume = resolve_resume_path(None, args.auto_resume, ckpt_dir, run_name)
    if resume is not None:
        start_epoch, best_iou, history = load_checkpoint(
            resume, model, optimizer, scheduler, scaler, device,
        )

    if start_epoch > args.epochs:
        print(f"Already done {args.epochs} epochs.")
    else:
        print(f"\nTraining {run_name}: epochs {start_epoch}..{args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, scaler, use_amp, args.modalities,
        )
        val_loss, val_m = validate(
            model, loaders["val"], criterion, device, use_amp, args.modalities,
        )
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | Train {train_loss:.4f} "
              f"(IoU {train_m['iou']:.3f}) | Val {val_loss:.4f} "
              f"(IoU {val_m['iou']:.3f}) | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_iou": train_m["iou"], **{f"val_{k}": v for k, v in val_m.items()},
            "lr": scheduler.get_last_lr()[0], "time": elapsed,
        })

        save_checkpoint(ckpt_dir / f"{run_name}_latest.pt",
                        epoch, model, optimizer, scheduler, scaler, best_iou,
                        history, extra={"val_iou": val_m["iou"]})
        save_history(history, log_dir / f"{run_name}_history.json")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_checkpoint(ckpt_dir / f"{run_name}_best.pt",
                            epoch, model, optimizer, scheduler, scaler, best_iou,
                            history, extra={"val_iou": val_m["iou"]})
            print(f"  -> new best ({best_iou:.4f})")

        if epoch % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"{run_name}_epoch{epoch}.pt",
                            epoch, model, optimizer, scheduler, scaler, best_iou,
                            history, extra={"val_iou": val_m["iou"]})

    # Final test eval
    ckpt = torch.load(ckpt_dir / f"{run_name}_best.pt",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_m = validate(model, loaders["test"], criterion, device, use_amp,
                         args.modalities)
    print(f"\nTest IoU: {test_m['iou']:.4f}  Dice: {test_m['dice']:.4f}")
    with open(log_dir / f"{run_name}_test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)


if __name__ == "__main__":
    main()
