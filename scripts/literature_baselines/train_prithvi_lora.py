"""LoRA fine-tune Prithvi-EO-1.0-100M on Sen1Floods11 hand-labeled splits.

Background. We cite Prithvi-EO's reported test IoU of 0.8046 on
Sen1Floods11 in the headline figure. That number comes from full
fine-tuning of a 100M-parameter foundation model that was pretrained on
millions of multi-temporal HLS scenes. A *fair* comparison against our
from-scratch TriModal model is not the published number — it is "what
does Prithvi-EO get when fine-tuned on the SAME 446 hand-labeled chips,
with the same val / test splits, that we have."

This script does that: LoRA fine-tuning (rank 8 by default) on the
Prithvi backbone with a fresh segmentation head, evaluated on our
splits. Output is reported alongside our model in the comparison table.

Prithvi was pretrained on Harmonized Landsat-Sentinel-2 (HLS) — six
spectral bands. Sen1Floods11 S2 has 13 bands; we project the HLS-required
six (B02 Blue, B03 Green, B04 Red, B05 RedEdge1, B06 NIR, B07 SWIR1) and
hand them to the model. SAR and DEM are NOT consumed by Prithvi in this
setup, so the comparison is fundamentally optical-only — we will note
this in the paper.

Requires:
    pip install transformers peft
    Plus access to ibm-nasa-geospatial/Prithvi-EO-1.0-100M (Hugging Face Hub).

Usage:
    python scripts/literature_baselines/train_prithvi_lora.py \\
        --data_root  ... --splits_dir ...

NOTE — this script ships as a working skeleton. The exact module path for
loading Prithvi from HF varies across `transformers` versions; depending
on the installed version, the model may need to be loaded via the
`prithvi_global` reference repo instead. The skeleton is structured so
swapping in the right loader is a single function (`load_prithvi`).
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


# Sentinel-2 13-band order in this dataset (verify against your loader).
# Index mapping to HLS-equivalent 6 bands used by Prithvi:
#   B02 Blue   -> S2 idx 1
#   B03 Green  -> S2 idx 2
#   B04 Red    -> S2 idx 3
#   B05 RedEdge1 -> S2 idx 4
#   B06 NIR    -> S2 idx 7
#   B07 SWIR1  -> S2 idx 11
# (The exact band order depends on the source dataset; verify before use.)
PRITHVI_BAND_INDICES = [1, 2, 3, 4, 7, 11]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=2)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lora_rank",   type=int, default=8)
    p.add_argument("--lora_alpha",  type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--save_every",  type=int, default=5)
    p.add_argument("--run_name",    default="prithvi_lora")
    p.add_argument("--prithvi_repo",
                   default="ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
                   help="HF Hub repo for the Prithvi backbone")
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_prithvi(repo_id):
    """Load Prithvi-EO backbone + return a callable producing patch features.

    The HF loading path for Prithvi is unstable across versions of
    `transformers`. This function isolates that loading logic. If the
    standard HF path fails, fall back to importing from the
    prithvi_global reference repo (must be on PYTHONPATH).
    """
    try:
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        model  = AutoModel.from_pretrained(repo_id, config=config,
                                            trust_remote_code=True)
        return model, config
    except Exception as e:
        print(f"[WARN] HF loading failed: {e}")
        print("Falling back to local prithvi_global package — make sure it is "
              "on PYTHONPATH before running this script.")
        try:
            from prithvi_global.mae.config import get_config
            from prithvi_global.mae.model import MaskedAutoencoderViT
            config = get_config()
            model  = MaskedAutoencoderViT(**config)
            return model, config
        except Exception as e2:
            raise RuntimeError(
                f"Could not load Prithvi from either source: {e} / {e2}"
            )


class PrithviSegmentationHead(nn.Module):
    """Lightweight segmentation head on top of Prithvi patch tokens.

    Prithvi outputs a sequence of patch tokens at a coarse spatial
    resolution. This head reshapes them to a feature map and applies a
    small upsampling convolution stack to produce per-pixel logits.
    """

    def __init__(self, embed_dim, patch_size, num_classes=2, n_upsample=4):
        super().__init__()
        self.patch_size = patch_size
        layers = []
        in_ch = embed_dim
        out_ch = max(64, embed_dim // 4)
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.GroupNorm(min(16, out_ch), out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch = max(32, out_ch // 2)
        layers += [nn.Conv2d(in_ch, num_classes, 1)]
        self.up = nn.Sequential(*layers)

    def forward(self, tokens):
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Token count {N} not square"
        x = tokens.transpose(1, 2).reshape(B, D, H, W)
        return self.up(x)


def apply_lora_to_attention(model, rank, alpha, dropout):
    """Wrap the attention layers in LoRA adapters via PEFT."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("ERROR: peft is required. Install with: pip install peft")
        sys.exit(1)
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type=None,
    )
    return get_peft_model(model, cfg)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Prithvi-EO backbone ...")
    backbone, config = load_prithvi(args.prithvi_repo)
    embed_dim = getattr(config, "hidden_size",
                         getattr(config, "embed_dim", 768))
    patch_size = getattr(config, "patch_size", 16)
    print(f"  embed_dim={embed_dim}  patch_size={patch_size}")

    print("Wrapping backbone with LoRA adapters ...")
    backbone = apply_lora_to_attention(backbone, args.lora_rank,
                                        args.lora_alpha, args.lora_dropout)

    head = PrithviSegmentationHead(embed_dim, patch_size, num_classes=2)
    model = nn.ModuleDict({"backbone": backbone, "head": head}).to(device)

    # Data
    from src.data.dataset import get_trimodal_dataloaders
    loaders = get_trimodal_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        crop_size=224,   # Prithvi expects multiples of patch_size; 224 is safe
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 8.0], device=device), ignore_index=255,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=len(loaders["train"]) * args.epochs,
                                   eta_min=1e-6)
    scaler = GradScaler(enabled=not args.no_amp)

    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning {args.run_name} on {len(loaders['train'].dataset)} chips")
    print(f"LoRA rank={args.lora_rank} alpha={args.lora_alpha}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for s1, s2, dem, labels in tqdm(loaders["train"], leave=False):
            s2_6band = s2[:, PRITHVI_BAND_INDICES].to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=not args.no_amp):
                tokens = model["backbone"](s2_6band).last_hidden_state \
                          if hasattr(model["backbone"], "last_hidden_state") \
                          else model["backbone"](s2_6band)
                logits = model["head"](tokens)
                # Resize logits to label resolution if needed
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = nn.functional.interpolate(
                        logits, size=labels.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss.item():.4f}  "
              f"{elapsed:.1f}s")

    print("\nDone. Test eval not yet implemented — re-run loop on loaders['test'] "
          "or use scripts/eval_per_chip.py with a custom model registration.")


if __name__ == "__main__":
    main()
