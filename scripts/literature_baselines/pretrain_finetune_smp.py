"""Two-phase weakly-labeled pretraining for SMP segmentation baselines.

Companion to scripts/pretrain_finetune_trimodal.py. Tests whether the
weak-pretrain regression observed for Fusion U-Net (-0.054 IoU) and
TriModal (-0.059 IoU) generalizes to non-cross-attention U-Net-family
architectures. Without this test, the "universal weak-pretrain
collapse" claim in the paper rests on only two cross-attention
architectures and a reviewer can correctly note that the regression
may be cross-attention specific.

Phase A: 30 epochs on ~4385 WeaklyLabeled chips with Otsu pseudo-labels.
Phase B: 60 epochs fine-tune on 446 HandLabeled chips at lr/5.
Eval:    HandLabeled test split (90 chips).

Input layout matches train_smp_baselines.py: modalities concatenated at
the input layer; SMP architecture's first conv accepts the full channel
count (S1 2 + S2 13 + DEM 2 = 17 channels for s1_s2_dem).

Usage on Isaac via slurm/pretrain_finetune_smp.sbatch. CLI mirrors the
TriModal weak-pretrain script for consistency.
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

from src.data.dataset import (
    get_trimodal_dataloaders,
    get_trimodal_weak_dataloaders,
)
from src.utils.metrics import MetricAccumulator
from src.utils.checkpoint import (
    save_checkpoint, load_checkpoint, resolve_resume_path, save_history,
)


MODALITY_CHANNELS = {"s1": 2, "s2": 13, "dem": 2}

SMP_BUILDERS = {
    "unet":           smp.Unet,
    "unetplusplus":   smp.UnetPlusPlus,
    "deeplabv3plus":  smp.DeepLabV3Plus,
    "fpn":            smp.FPN,
    "manet":          smp.MAnet,
}


def parse_args():
    p = argparse.ArgumentParser()

    # Architecture / modality
    p.add_argument("--arch",       default="unetplusplus", choices=list(SMP_BUILDERS))
    p.add_argument("--encoder",    default="resnet34")
    p.add_argument("--encoder_weights", default="imagenet",
                   help="Encoder pretraining; set to 'none' to disable")
    p.add_argument("--modalities", default="s1_s2_dem",
                   help="Underscore-joined modality keys; concatenated at input")

    # HandLabeled paths (used for fine-tune + final test eval)
    p.add_argument("--hand_data_root",   required=True)
    p.add_argument("--hand_splits_dir",  required=True)

    # WeaklyLabeled paths (used for pretrain)
    p.add_argument("--weak_data_root",   required=True)
    p.add_argument("--weak_splits_dir",  required=True)
    p.add_argument("--weak_train_csv",   default="flood_train_data.csv")
    p.add_argument("--weak_val_csv",     default="flood_valid_data.csv")
    p.add_argument("--weak_s1_subdir",   default="S1Weak")
    p.add_argument("--weak_s2_subdir",   default="S2Weak")
    p.add_argument("--weak_dem_subdir",  default="DEMWeak")
    p.add_argument("--weak_label_subdir", default="S1OtsuLabelWeak")
    p.add_argument("--weak_s1_token",    default="S1Weak")
    p.add_argument("--weak_s2_token",    default="S2Weak")
    p.add_argument("--weak_dem_token",   default="DEMWeak")

    # Output
    p.add_argument("--output_dir",       default="results")
    p.add_argument("--run_name",         default=None)

    # Phase A (pretrain on weak)
    p.add_argument("--pretrain_epochs",     type=int,   default=30)
    p.add_argument("--pretrain_lr",         type=float, default=2e-4)
    p.add_argument("--pretrain_batch_size", type=int,   default=4)

    # Phase B (fine-tune on hand)
    p.add_argument("--finetune_epochs",     type=int,   default=60)
    p.add_argument("--finetune_lr",         type=float, default=4e-5)
    p.add_argument("--finetune_batch_size", type=int,   default=4)

    # Shared
    p.add_argument("--crop_size",   type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="auto")
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--save_every",  type=int, default=5)
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--skip_pretrain", action="store_true")
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_device(arg):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_smp_model(arch, in_channels, encoder, encoder_weights):
    builder = SMP_BUILDERS[arch]
    return builder(
        encoder_name=encoder,
        encoder_weights=None if encoder_weights == "none" else encoder_weights,
        in_channels=in_channels,
        classes=2,
    )


def select_and_concat(s1, s2, dem, modalities):
    pool = {"s1": s1, "s2": s2, "dem": dem}
    keys = modalities.split("_")
    return torch.cat([pool[k] for k in keys], dim=1)


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, scaler, use_amp, modalities):
    model.train()
    total_loss, n = 0.0, 0
    acc = MetricAccumulator()
    pbar = tqdm(loader, desc="  Train", leave=False)
    for s1, s2, dem, labels in pbar:
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
        x = select_and_concat(s1, s2, dem, modalities)
        with autocast(device_type="cuda", enabled=use_amp):
            out  = model(x)
            loss = criterion(out, labels)
        if not torch.isnan(loss):
            total_loss += loss.item(); n += 1
        acc.update(torch.argmax(out, dim=1), labels, ignore_index=255)
    return total_loss / max(n, 1), acc.compute()


def run_phase(model, loaders, device, scaler, args, *,
              phase, epochs, lr, run_tag, ckpt_dir, log_dir, use_amp,
              run_name):
    class_weights = torch.tensor([1.0, 8.0], device=device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer     = AdamW(model.parameters(), lr=lr)
    scheduler     = CosineAnnealingLR(optimizer,
                                      T_max=len(loaders["train"]) * epochs,
                                      eta_min=1e-6)

    full_run = f"{run_name}_{run_tag}"

    start_epoch, best_iou, history = 1, 0.0, []
    if args.auto_resume:
        resume_path = resolve_resume_path(None, True, ckpt_dir, full_run)
        if resume_path is not None:
            start_epoch, best_iou, history = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device,
            )

    if start_epoch > epochs:
        print(f"[{phase}] already completed {epochs} epochs — skipping.")
        return best_iou

    print(f"\n[{phase}] training epochs {start_epoch}..{epochs}  lr={lr}")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            device, scaler, use_amp, args.modalities,
        )
        val_loss, val_m = validate(
            model, loaders["val"], criterion, device, use_amp, args.modalities,
        )
        elapsed = time.time() - t0
        iou_gap = train_m["iou"] - val_m["iou"]
        print(f"[{phase}] {epoch:3d}/{epochs} | "
              f"Train {train_loss:.4f} (IoU {train_m['iou']:.3f}) | "
              f"Val {val_loss:.4f} (IoU {val_m['iou']:.3f}) | "
              f"Gap {iou_gap:+.3f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "phase": phase,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_iou": train_m["iou"], "train_dice": train_m["dice"],
            **{f"val_{k}": v for k, v in val_m.items()},
            "iou_gap": iou_gap,
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
        })

        save_checkpoint(
            ckpt_dir / f"{full_run}_latest.pt",
            epoch, model, optimizer, scheduler, scaler, best_iou, history,
            extra={"val_iou": val_m["iou"]},
        )
        save_history(history, log_dir / f"{full_run}_history.json")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            save_checkpoint(
                ckpt_dir / f"{full_run}_best.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"]},
            )
            print(f"  -> [{phase}] new best val IoU: {best_iou:.4f}")

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"{full_run}_epoch{epoch}.pt",
                epoch, model, optimizer, scheduler, scaler, best_iou, history,
                extra={"val_iou": val_m["iou"]},
            )
    return best_iou


def main():
    args = parse_args()
    use_amp = not args.no_amp
    set_seed(args.seed)
    device = get_device(args.device)

    in_channels = sum(MODALITY_CHANNELS[m] for m in args.modalities.split("_"))
    run_name = args.run_name or (
        f"smp_{args.arch}_{args.encoder}_{args.modalities}_weakpretrain"
    )

    print(f"Device: {device}  AMP: {use_amp}")
    print(f"Building {run_name}: {args.arch}/{args.encoder} "
          f"in_ch={in_channels} weights={args.encoder_weights}")

    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    log_dir    = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = build_smp_model(args.arch, in_channels,
                            args.encoder, args.encoder_weights).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    scaler = GradScaler(enabled=use_amp)

    # ----------------------------------------------------------------------
    # Phase A: pretrain on WeaklyLabeled
    # ----------------------------------------------------------------------
    if not args.skip_pretrain:
        print("\n--- Phase A: pretrain on WeaklyLabeled ---")
        weak_loaders = get_trimodal_weak_dataloaders(
            data_root=args.weak_data_root,
            splits_dir=args.weak_splits_dir,
            batch_size=args.pretrain_batch_size,
            num_workers=args.num_workers,
            crop_size=args.crop_size,
            s1_subdir=args.weak_s1_subdir,
            s2_subdir=args.weak_s2_subdir,
            dem_subdir=args.weak_dem_subdir,
            label_subdir=args.weak_label_subdir,
            s1_token=args.weak_s1_token,
            s2_token=args.weak_s2_token,
            dem_token=args.weak_dem_token,
            train_csv=args.weak_train_csv,
            val_csv=args.weak_val_csv,
        )
        print(f"  Weak train: {len(weak_loaders['train'].dataset)} samples")
        print(f"  Weak val:   {len(weak_loaders['val'].dataset)} samples")

        pretrain_best = run_phase(
            model, weak_loaders, device, scaler, args,
            phase="PRETRAIN",
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            run_tag="pretrain",
            ckpt_dir=ckpt_dir, log_dir=log_dir, use_amp=use_amp,
            run_name=run_name,
        )
        print(f"\nPhase A best (weak val) IoU: {pretrain_best:.4f}")
    else:
        print("--skip_pretrain set — loading existing _pretrain_best.pt for Phase B")

    # ----------------------------------------------------------------------
    # Phase B: fine-tune on HandLabeled
    # ----------------------------------------------------------------------
    pretrain_best_path = ckpt_dir / f"{run_name}_pretrain_best.pt"
    if not pretrain_best_path.exists():
        raise FileNotFoundError(
            f"Pretrain best checkpoint not found at {pretrain_best_path}. "
            "Run phase A first, or supply a custom path."
        )
    print(f"\nLoading pretrained weights from {pretrain_best_path}")
    ckpt = torch.load(pretrain_best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n--- Phase B: fine-tune on HandLabeled ---")
    hand_loaders = get_trimodal_dataloaders(
        data_root=args.hand_data_root,
        splits_dir=args.hand_splits_dir,
        batch_size=args.finetune_batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
    )
    print(f"  Hand train: {len(hand_loaders['train'].dataset)} samples")
    print(f"  Hand val:   {len(hand_loaders['val'].dataset)} samples")
    print(f"  Hand test:  {len(hand_loaders['test'].dataset)} samples")

    finetune_best = run_phase(
        model, hand_loaders, device, scaler, args,
        phase="FINETUNE",
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        run_tag="finetune",
        ckpt_dir=ckpt_dir, log_dir=log_dir, use_amp=use_amp,
        run_name=run_name,
    )

    # ----------------------------------------------------------------------
    # Final test eval on HandLabeled test split
    # ----------------------------------------------------------------------
    print("\n--- Test evaluation (HandLabeled test split, fine-tuned model) ---")
    finetune_best_path = ckpt_dir / f"{run_name}_finetune_best.pt"
    ckpt = torch.load(finetune_best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 8.0], device=device), ignore_index=255,
    )
    _, test_m = validate(model, hand_loaders["test"], test_criterion,
                          device, use_amp, args.modalities)

    print(f"Test IoU:  {test_m['iou']:.4f}")
    print(f"Test Dice: {test_m['dice']:.4f}")
    print(f"Test Prec: {test_m['precision']:.4f}")
    print(f"Test Rec:  {test_m['recall']:.4f}")

    out_path = log_dir / f"{run_name}_finetune_test_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "pretrain_best_val_iou": float(ckpt.get("val_iou", finetune_best)),
            "test_metrics": test_m,
            "args": vars(args),
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
