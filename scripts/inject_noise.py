"""Robustness evaluation: inject realistic noise / corruption into inputs
at inference time and measure how IoU degrades.

Three perturbation modes covered:

  --mode cloud     Mask N% of S2 pixels with cloud-shaped occlusions.
                   Realistic operational scenario: mid-cloud fronts after
                   a flood event saturate part of the optical scene.

  --mode speckle   Apply multiplicative Gamma noise to S1 (mean=1, var=σ²).
                   Sentinel-1 ground-range-detected products already have
                   speckle; this stresses the model with stronger speckle.

  --mode dem_noise Add Gaussian noise + a small constant bias to DEM
                   elevation. Models GLO-30 vs SRTM disagreement and
                   imperfect bilinear DEM resampling.

Each mode sweeps a severity parameter; results table is saved as JSON +
markdown + a plot of IoU vs severity.

Usage:
    python scripts/inject_noise.py \\
        --mode cloud --severities 0 0.1 0.3 0.5 0.7 0.9 \\
        --model trimodal --checkpoint results/checkpoints/trimodal_unet_best.pt \\
        --data_root ... --splits_dir ...
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import MetricAccumulator


# --------------------------------------------------------------------------
# Perturbation kernels — applied to a (B, C, H, W) tensor; in-place safe
# because we always work on a clone of the original input.
# --------------------------------------------------------------------------

def cloud_mask(s2, fraction, seed):
    """Replace `fraction` of pixels (cloud-shaped) with the per-band mean.

    We use a smooth random mask (Gaussian-blurred uniform noise then
    thresholded) rather than i.i.d. pixel dropout, because real clouds
    are spatially coherent. The mask is shared across all S2 bands.
    """
    if fraction <= 0:
        return s2
    B, C, H, W = s2.shape
    rng = np.random.default_rng(seed)

    # Generate a smooth random field per chip.
    masks = []
    for _ in range(B):
        noise = rng.standard_normal((H, W)).astype(np.float32)
        # Cheap "Gaussian blur" via a 5-iteration boxcar — keeps deps minimal
        kernel = np.ones((9, 9), dtype=np.float32) / 81.0
        from scipy.signal import convolve2d
        smoothed = noise
        for _ in range(2):
            smoothed = convolve2d(smoothed, kernel, mode="same", boundary="symm")
        # Threshold so that the desired fraction of pixels becomes "cloud"
        thresh = np.quantile(smoothed, 1 - fraction)
        masks.append((smoothed >= thresh).astype(np.float32))
    mask = torch.from_numpy(np.stack(masks)).to(s2.device)  # (B, H, W)

    # Replace masked pixels with the per-band mean (the dataset is
    # pre-normalized, so band-mean is approximately zero — choosing 0 here
    # rather than the true band-mean keeps the script simple).
    fill = torch.zeros_like(s2)
    return torch.where(mask.unsqueeze(1).bool(), fill, s2)


def sar_speckle(s1, sigma, seed):
    """Multiplicative Gamma-like noise: x' = x * (1 + eps), eps ~ N(0, sigma)."""
    if sigma <= 0:
        return s1
    rng   = np.random.default_rng(seed)
    eps   = rng.normal(0.0, sigma, size=s1.shape).astype(np.float32)
    eps_t = torch.from_numpy(eps).to(s1.device)
    return s1 * (1.0 + eps_t)


def dem_noise(dem, sigma, bias, seed):
    """Gaussian elevation noise + constant bias on band-0 of DEM."""
    if sigma <= 0 and bias == 0:
        return dem
    rng = np.random.default_rng(seed)
    out = dem.clone()
    # Only perturb the elevation band (band 0), leave slope (band 1) alone.
    # The slope was computed offline, so a clean comparison is simpler if
    # we don't recompute it under noise.
    if sigma > 0:
        n = rng.normal(0.0, sigma, size=dem[:, 0].shape).astype(np.float32)
        out[:, 0] = out[:, 0] + torch.from_numpy(n).to(out.device)
    if bias != 0:
        out[:, 0] = out[:, 0] + float(bias)
    return out


# --------------------------------------------------------------------------
# Model dispatch
# --------------------------------------------------------------------------

def build_model(model_kind, modalities=None):
    if model_kind == "trimodal":
        from src.models.trimodal_unet import TriModalFusionUNet
        return TriModalFusionUNet()
    if model_kind == "fusion":
        from src.models.fusion_unet import FusionUNet
        return FusionUNet()
    if model_kind == "bimodal":
        from src.models.bimodal_cross_attn_unet import build_bimodal
        return build_bimodal(tuple(modalities.split("_")))
    raise ValueError(model_kind)


def model_forward(model, kind, modalities, s1, s2, dem, device, use_amp):
    pool = {"s1": s1.to(device), "s2": s2.to(device), "dem": dem.to(device)}
    with autocast(device_type="cuda", enabled=use_amp):
        if kind == "trimodal":
            return model(pool["s1"], pool["s2"], pool["dem"])
        if kind == "fusion":
            return model(pool["s1"], pool["s2"])
        if kind == "bimodal":
            a, b = modalities.split("_")
            return model(pool[a], pool[b])
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       required=True,
                   choices=["cloud", "speckle", "dem_noise"])
    p.add_argument("--severities", nargs="+", type=float, required=True,
                   help="Severity values to sweep (mode-specific units)")
    p.add_argument("--dem_bias",   type=float, default=0.0,
                   help="For mode=dem_noise: constant bias added in addition to sigma")
    p.add_argument("--model",      required=True,
                   choices=["trimodal", "fusion", "bimodal"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--modalities", default=None)
    p.add_argument("--data_root",  required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--split",      default="test", choices=["val", "test"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no_amp",     action="store_true")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--run_name",   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, args.modalities).to(device).eval()
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    from src.data.dataset import get_trimodal_dataloaders
    loaders = get_trimodal_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=1, num_workers=0, crop_size=None,
    )
    loader = loaders[args.split]
    print(f"Mode={args.mode}  Sweep={args.severities}  Split={args.split}  "
          f"Chips={len(loader)}")

    rows = []
    with torch.no_grad():
        for severity in args.severities:
            acc = MetricAccumulator()
            for s1, s2, dem, label in tqdm(loader, desc=f"  sev={severity}"):
                s1c, s2c, demc = s1.clone(), s2.clone(), dem.clone()
                # Apply the perturbation
                if args.mode == "cloud":
                    s2c = cloud_mask(s2c, severity, seed=args.seed)
                elif args.mode == "speckle":
                    s1c = sar_speckle(s1c, severity, seed=args.seed)
                elif args.mode == "dem_noise":
                    demc = dem_noise(demc, severity, args.dem_bias,
                                     seed=args.seed)

                logits = model_forward(model, args.model, args.modalities,
                                        s1c, s2c, demc, device, use_amp)
                pred   = torch.argmax(logits, dim=1)
                acc.update(pred.cpu(), label.cpu(), ignore_index=255)

            m = acc.compute()
            row = {"severity": float(severity), **{k: float(m[k]) for k in
                   ("iou", "dice", "precision", "recall", "accuracy")}}
            rows.append(row)
            print(f"    sev={severity}  IoU={row['iou']:.4f}  "
                  f"Dice={row['dice']:.4f}")

    run_name = args.run_name or (
        f"{Path(args.checkpoint).stem.replace('_best','')}_{args.mode}"
    )
    log_dir  = Path(args.output_dir) / "logs"
    fig_dir  = Path(args.output_dir) / "figures"
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_json = log_dir / f"{run_name}_robust.json"
    with open(out_json, "w") as f:
        json.dump({
            "mode": args.mode, "checkpoint": str(args.checkpoint),
            "split": args.split, "rows": rows,
        }, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Quick plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sev = [r["severity"] for r in rows]
        iou = [r["iou"]      for r in rows]
        ax.plot(sev, iou, "o-", color="#3182bd", linewidth=2, markersize=8)
        ax.set_xlabel(f"{args.mode} severity")
        ax.set_ylabel("Test IoU")
        ax.set_title(f"Robustness: {args.mode} ({run_name})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_png = fig_dir / f"{run_name}_robust.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")


if __name__ == "__main__":
    main()
