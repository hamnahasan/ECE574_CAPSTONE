"""Post-hoc temperature scaling — Guo et al. (2017) calibration baseline.

A single scalar T is fit on the validation set by minimizing NLL of the
softmax(logits / T). At inference, the same T is divided into the test-time
logits before softmax. T > 1 reduces overconfidence; T < 1 increases it.

This is the simplest and most widely-deployed post-hoc calibration method
in the literature. Reviewers will expect to see how MC Dropout compares
against it. Expected outcome on our model:

    pre-temperature  ECE = 0.0273  (already well-calibrated, so little room)
    post-temperature ECE = 0.020-0.025

If MC Dropout already lands inside the well-calibrated band, temperature
scaling will yield only marginal gains — that is itself a useful result.

Usage:
    python scripts/temperature_scaling.py \\
        --model trimodal \\
        --checkpoint results/checkpoints/trimodal_unet_best.pt \\
        --data_root  ... --splits_dir ...
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.uncertainty import compute_ece


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=["trimodal", "fusion", "bimodal"])
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--modalities",  default=None)
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--n_bins",      type=int, default=15)
    p.add_argument("--max_iter",    type=int, default=200,
                   help="Max LBFGS iterations to fit T")
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--run_name",    default=None)
    p.add_argument("--no_amp",      action="store_true")
    return p.parse_args()


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


@torch.no_grad()
def collect_logits(model, loader, kind, modalities, device, use_amp):
    """Run the model once over a loader and stack (logits, labels) per pixel."""
    all_logits = []
    all_labels = []
    for s1, s2, dem, label in tqdm(loader, desc="    Collecting logits"):
        logits = model_forward(model, kind, modalities, s1, s2, dem,
                                device, use_amp)
        # logits: (1, 2, H, W) -> reshape to (H*W, 2). Drop nodata pixels.
        l = logits.squeeze(0).permute(1, 2, 0).reshape(-1, 2).float().cpu()
        y = label.squeeze(0).reshape(-1).cpu()
        valid = y != 255
        all_logits.append(l[valid])
        all_labels.append(y[valid])
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(logits, labels, max_iter=200):
    """Fit a single T by minimizing NLL on (logits, labels) using LBFGS."""
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    crit = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = crit(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def ece_at_temperature(logits, labels, T, n_bins):
    """Evaluate ECE on a (logits, labels) pair at the given T."""
    probs = TF.softmax(logits / T, dim=1)[:, 1].numpy()
    return compute_ece(probs, labels.numpy(), n_bins=n_bins)[0]


def main():
    args = parse_args()
    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build / load model
    model = build_model(args.model, args.modalities).to(device).eval()
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Data: val for fitting T, test for reporting
    from src.data.dataset import get_trimodal_dataloaders
    loaders = get_trimodal_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=1, num_workers=0, crop_size=None,
    )

    print("Collecting logits on val (used to fit T)...")
    val_logits, val_labels = collect_logits(
        model, loaders["val"], args.model, args.modalities, device, use_amp,
    )
    print("Collecting logits on test (used to report ECE)...")
    test_logits, test_labels = collect_logits(
        model, loaders["test"], args.model, args.modalities, device, use_amp,
    )

    # Fit T on val
    print("\nFitting temperature on val ...")
    T = fit_temperature(val_logits, val_labels.long(), max_iter=args.max_iter)
    print(f"  T* = {T:.4f}")

    # Pre/post ECE on test
    ece_pre  = ece_at_temperature(test_logits, test_labels, T=1.0, n_bins=args.n_bins)
    ece_post = ece_at_temperature(test_logits, test_labels, T=T,   n_bins=args.n_bins)
    print(f"  Test ECE pre-temperature : {ece_pre:.4f}")
    print(f"  Test ECE post-temperature: {ece_post:.4f}  "
          f"(delta {ece_post - ece_pre:+.4f})")

    run_name = args.run_name or Path(args.checkpoint).stem.replace("_best", "")
    log_dir  = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{run_name}_temperature.json"
    with open(out, "w") as f:
        json.dump({
            "checkpoint":     str(args.checkpoint),
            "T_optimal":      T,
            "ece_pre":        ece_pre,
            "ece_post":       ece_post,
            "delta_ece":      ece_post - ece_pre,
            "n_bins":         args.n_bins,
        }, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
