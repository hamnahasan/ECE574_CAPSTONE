"""Render presentation-quality prediction galleries — runs on Isaac.

Produces three figures for the paper / presentation:

  1. results/figures/paper_s1s2_ablation_predictions.png
     S1+S2 early-fusion best ablation (0.7991 test IoU) — best 3 + worst 3 chips
     on the test split. Shows what the strongest model gets right and where it
     still fails.

  2. results/figures/paper_trimodal_bolivia_predictions.png
     TriModal U-Net on Bolivia held-out — best 3 + worst 3 chips. The
     cross-region generalization story made visual.

  3. results/figures/paper_disagreement_bolivia_103757.png
     Bolivia_103757 across all 3 primary models. FCN nails it (0.79 IoU);
     Fusion (0.09) and TriModal (0.05) collapse because the S2 imagery is
     cloud-contaminated and poisons the fusion signal. This is the smoking-gun
     example for slide 11 (root-cause error analysis).

Why this script lives on Isaac:
  - Needs S1, S2, DEM, Label rasters from /lustre — only available on Isaac
  - Needs the trained checkpoints — also on Isaac in results/checkpoints/
  - Needs GPU for inference (especially trimodal, 70M params)

Usage on Isaac:
    cd /lustre/isaac24/scratch/$USER/ECE574_CAPSTONE
    module load cuda/12.1.1-binary
    eval "$(conda shell.bash hook)"
    conda activate floodseg
    python scripts/render_paper_figures.py

Then rsync results/figures/paper_*.png back locally.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    Sen1Floods11, Sen1Floods11MultiModal, Sen1Floods11TriModal,
    S1_MEAN, S1_STD, S2_MEAN, S2_STD, DEM_MEAN, DEM_STD,
)


# ---------------------------------------------------------------------------
# Defaults — override via CLI
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT  = "/lustre/isaac24/scratch/{user}/sen1flood1/v1.1/data/flood_events/HandLabeled"
DEFAULT_SPLITS_DIR = "/lustre/isaac24/scratch/{user}/sen1flood1/v1.1/splits/flood_handlabeled"
DEFAULT_CKPT_DIR   = "results/checkpoints"
DEFAULT_LOGS_DIR   = "results/logs"
DEFAULT_FIG_DIR    = "results/figures"

DISAGREEMENT_CHIP = "Bolivia_103757_S1Hand.tif"  # Slide 11 smoking-gun example


def parse_args():
    user = os.environ.get("USER", "")
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default=DEFAULT_DATA_ROOT.format(user=user))
    p.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR.format(user=user))
    p.add_argument("--ckpt_dir",   default=DEFAULT_CKPT_DIR)
    p.add_argument("--logs_dir",   default=DEFAULT_LOGS_DIR)
    p.add_argument("--fig_dir",    default=DEFAULT_FIG_DIR)
    p.add_argument("--device",     default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def s2_to_rgb(s2_tensor):
    """Convert a normalized 13-band S2 tensor to a displayable true-color RGB.

    Sentinel-2 band order in Sen1Floods11: [B1, B2, B3, B4, B5, B6, B7, B8,
    B8A, B9, B10, B11, B12]. True color = (B4, B3, B2) = indices (3, 2, 1).
    """
    s2 = s2_tensor.cpu().numpy()
    # Denormalize
    mean = np.array(S2_MEAN).reshape(-1, 1, 1)
    std  = np.array(S2_STD).reshape(-1, 1, 1)
    s2_denorm = s2 * std + mean
    rgb = np.stack([s2_denorm[3], s2_denorm[2], s2_denorm[1]], axis=-1)  # (H, W, 3)
    # Stretch to [0, 1] for display (Sen1Floods11 reflectance values are typically <0.4)
    rgb = np.clip(rgb / 0.3, 0, 1)
    return rgb


def s1_to_grayscale(s1_tensor):
    """Convert normalized 2-band S1 tensor to displayable grayscale (VV channel)."""
    s1 = s1_tensor.cpu().numpy()
    mean = np.array(S1_MEAN).reshape(-1, 1, 1)
    std  = np.array(S1_STD).reshape(-1, 1, 1)
    s1_denorm = s1 * std + mean
    vv = s1_denorm[0]  # VV channel
    # Sen1Floods11 stores normalized backscatter; stretch for display
    return np.clip((vv - 0.3) / 0.6, 0, 1)


def dem_to_grayscale(dem_tensor):
    """Convert normalized 2-band DEM tensor to displayable grayscale (elevation)."""
    dem = dem_tensor.cpu().numpy()
    mean = np.array(DEM_MEAN).reshape(-1, 1, 1)
    std  = np.array(DEM_STD).reshape(-1, 1, 1)
    elev = dem[0] * std[0] + mean[0]
    # Per-chip min-max stretch (elevation range varies wildly by region)
    if elev.max() > elev.min():
        return (elev - elev.min()) / (elev.max() - elev.min())
    return np.zeros_like(elev)


def error_overlay(pred, label):
    """Build an RGB error map: TP=green, FP=red, FN=blue, TN=light gray."""
    h, w = pred.shape
    rgb = np.full((h, w, 3), 0.92, dtype=np.float32)  # light gray TN
    valid = label != 255
    tp = valid & (pred == 1) & (label == 1)
    fp = valid & (pred == 1) & (label == 0)
    fn = valid & (pred == 0) & (label == 1)
    rgb[tp] = [0.20, 0.70, 0.30]   # green
    rgb[fp] = [0.85, 0.15, 0.15]   # red
    rgb[fn] = [0.20, 0.30, 0.85]   # blue
    return rgb


def label_to_display(label):
    """Convert label (0/1/255) to a display image with nodata as gray."""
    label = label.astype(float)
    disp = np.where(label == 255, 0.5, label)
    return disp


# ---------------------------------------------------------------------------
# Per-chip data loading
# ---------------------------------------------------------------------------
def ensure_test_perchip(model_name, modalities, ckpt_path, args):
    """Make sure a per-chip JSON exists for this model on test. Re-run eval if not."""
    out_json = Path(args.logs_dir) / f"{model_name}_test_results.json"
    if out_json.exists():
        with open(out_json) as f:
            data = json.load(f)
        if isinstance(data, dict) and "per_chip" in data:
            return out_json
        print(f"  {out_json.name} exists but has no per_chip array — re-running eval")

    print(f"  Re-running evaluate.py for {model_name} on test split...")
    cmd = [
        "python", "scripts/evaluate.py",
        "--model",     "ablation" if modalities else "fcn" if model_name == "fcn_baseline" else "fusion" if model_name == "fusion_unet" else "trimodal",
        "--checkpoint", str(ckpt_path),
        "--data_root",  args.data_root,
        "--splits_dir", args.splits_dir,
        "--split",      "test",
        "--output",     str(out_json),
    ]
    if modalities:
        cmd += ["--modalities", modalities]
    subprocess.run(cmd, check=True)
    return out_json


def best_and_worst_chips(perchip_json, k=3):
    """Return (best_k_chip_files, worst_k_chip_files) sorted by IoU."""
    with open(perchip_json) as f:
        data = json.load(f)
    chips = data["per_chip"]
    chips_sorted = sorted(chips, key=lambda c: c["iou"], reverse=True)
    best = chips_sorted[:k]
    worst = chips_sorted[-k:][::-1]  # worst-first
    return best, worst


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_fcn(ckpt_path, device):
    from src.models.fcn_baseline import FCNBaseline
    model = FCNBaseline(in_channels=2, num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model


def load_fusion(ckpt_path, device):
    from src.models.fusion_unet import FusionUNet
    model = FusionUNet(s1_channels=2, s2_channels=13, num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model


def load_trimodal(ckpt_path, device):
    from src.models.trimodal_unet import TriModalFusionUNet
    model = TriModalFusionUNet(s1_channels=2, s2_channels=13,
                                dem_channels=2, num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model


def load_ablation(ckpt_path, modalities, device):
    from src.models.early_fusion_unet import build_early_fusion
    model, _ = build_early_fusion(modalities, num_classes=2)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_fcn(model, s1, device):
    return torch.argmax(model(s1.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_fusion(model, s1, s2, device):
    return torch.argmax(model(s1.unsqueeze(0).to(device),
                                s2.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_trimodal(model, s1, s2, dem, device):
    return torch.argmax(model(s1.unsqueeze(0).to(device),
                                s2.unsqueeze(0).to(device),
                                dem.unsqueeze(0).to(device)), dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_ablation_s1s2(model, s1, s2, device):
    x = torch.cat([s1, s2], dim=0).unsqueeze(0).to(device)
    return torch.argmax(model(x), dim=1).squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Dataset access helpers (return tensors for one chip by file name)
# ---------------------------------------------------------------------------
def trimodal_dataset(data_root, splits_dir, split):
    split_map = {"test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}
    return Sen1Floods11TriModal(
        split_csv=Path(splits_dir) / split_map[split],
        s1_dir=Path(data_root) / "S1Hand",
        s2_dir=Path(data_root) / "S2Hand",
        dem_dir=Path(data_root) / "DEMHand",
        label_dir=Path(data_root) / "LabelHand",
        crop_size=None, augment=False, normalize=True,
    )


def find_chip_index(dataset, chip_filename):
    """Locate a chip by S1 filename — returns index or None."""
    for i, sample in enumerate(dataset.samples):
        if sample[0] == chip_filename:
            return i
    return None


# ---------------------------------------------------------------------------
# Figure 1 + 2: 4-panel best/worst gallery
# ---------------------------------------------------------------------------
def render_gallery(title, chip_records, dataset, predict_fn, save_path):
    """Render a 6-row × 4-col gallery (best 3 then worst 3).

    chip_records: list of {file, iou} dicts (best then worst, length 6)
    predict_fn:   callable(dataset_sample) -> pred numpy array
    """
    n = len(chip_records)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    col_titles = ["S2 (RGB)", "Ground Truth", "Prediction", "Errors (TP=green, FP=red, FN=blue)"]

    for row, rec in enumerate(chip_records):
        idx = find_chip_index(dataset, rec["file"])
        if idx is None:
            print(f"  [warn] chip not found in dataset: {rec['file']}")
            continue
        sample = dataset[idx]
        s1, s2, dem, label = sample
        pred = predict_fn(sample)
        label_np = label.numpy()

        axes[row, 0].imshow(s2_to_rgb(s2))
        axes[row, 1].imshow(label_to_display(label_np), cmap="Blues", vmin=0, vmax=1)
        axes[row, 2].imshow(pred,                       cmap="Blues", vmin=0, vmax=1)
        axes[row, 3].imshow(error_overlay(pred, label_np))

        for col, t in enumerate(col_titles):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(t, fontsize=11)

        chip_short = rec["file"].replace("_S1Hand.tif", "")
        axes[row, 0].set_ylabel(f"{chip_short}\nIoU={rec['iou']:.3f}",
                                  fontsize=10, rotation=0, ha="right", va="center", labelpad=50)

    fig.suptitle(title, fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: Bolivia_103757 cross-model disagreement (slide 11 example)
# ---------------------------------------------------------------------------
def render_disagreement(args, device, save_path):
    """7-panel single-row figure: S2 RGB | S1 VV | DEM | GT | FCN | Fusion | TriModal.

    Bolivia_103757 — FCN gets 0.79 IoU, Fusion 0.09, TriModal 0.05.
    The S2 imagery on this chip is cloud-contaminated; fusion models trust the
    S2 signal too much and collapse. The S1-only model is unaffected.
    """
    ds = trimodal_dataset(args.data_root, args.splits_dir, "bolivia")
    idx = find_chip_index(ds, DISAGREEMENT_CHIP)
    if idx is None:
        print(f"  [warn] {DISAGREEMENT_CHIP} not in Bolivia split — skipping disagreement figure")
        return
    s1, s2, dem, label = ds[idx]
    label_np = label.numpy()

    # Run each model
    fcn = load_fcn(Path(args.ckpt_dir) / "fcn_baseline_best.pt", device)
    pred_fcn = predict_fcn(fcn, s1, device)
    del fcn

    fus = load_fusion(Path(args.ckpt_dir) / "fusion_unet_best.pt", device)
    pred_fus = predict_fusion(fus, s1, s2, device)
    del fus

    tri = load_trimodal(Path(args.ckpt_dir) / "trimodal_unet_best.pt", device)
    pred_tri = predict_trimodal(tri, s1, s2, dem, device)
    del tri

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 7, figsize=(22, 4))
    panels = [
        ("S2 (RGB)",                       s2_to_rgb(s2),                        None, None),
        ("S1 (VV backscatter)",            s1_to_grayscale(s1),                  "gray", None),
        ("DEM (elevation)",                dem_to_grayscale(dem),                "terrain", None),
        ("Ground Truth",                   label_to_display(label_np),           "Blues", (0, 1)),
        (f"FCN pred (IoU=0.79)",           pred_fcn,                             "Blues", (0, 1)),
        (f"Fusion pred (IoU=0.09)",        pred_fus,                             "Blues", (0, 1)),
        (f"TriModal pred (IoU=0.05)",      pred_tri,                             "Blues", (0, 1)),
    ]
    for ax, (t, img, cmap, vrange) in zip(axes, panels):
        if cmap is None:
            ax.imshow(img)
        elif vrange is not None:
            ax.imshow(img, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(t, fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Bolivia_103757 — cloud-contaminated S2 poisons fusion: "
        "S1-only baseline (0.79 IoU) wins; multi-modal models collapse",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    # ----------------------------------------------------------------------
    # Figure 1: ablation_s1_s2 best/worst on test
    # ----------------------------------------------------------------------
    print("\n=== Figure 1: ablation_s1_s2 best/worst on test ===")
    # Ablation training saves checkpoints in a per-run subdirectory:
    #   results/checkpoints/ablation_s1_s2/ablation_s1_s2_best.pt
    # (Tri-modal/fusion training saves directly under results/checkpoints/.)
    s1s2_ckpt = Path(args.ckpt_dir) / "ablation_s1_s2" / "ablation_s1_s2_best.pt"
    if not s1s2_ckpt.exists():
        # Fall back to flat layout in case anyone moved it
        flat = Path(args.ckpt_dir) / "ablation_s1_s2_best.pt"
        if flat.exists():
            s1s2_ckpt = flat
    if not s1s2_ckpt.exists():
        print(f"  [skip] checkpoint not found: {s1s2_ckpt}")
    else:
        perchip_json = ensure_test_perchip("ablation_s1_s2", "s1_s2", s1s2_ckpt, args)
        best, worst = best_and_worst_chips(perchip_json, k=3)
        ds = trimodal_dataset(args.data_root, args.splits_dir, "test")
        model = load_ablation(s1s2_ckpt, "s1_s2", device)
        def pred_fn(sample):
            s1, s2, dem, _ = sample
            return predict_ablation_s1s2(model, s1, s2, device)
        render_gallery(
            "S1+S2 Early Fusion (best ablation, 0.7991 test IoU) — Best 3 (top) vs Worst 3 (bottom)",
            best + worst, ds, pred_fn,
            Path(args.fig_dir) / "paper_s1s2_ablation_predictions.png",
        )
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Figure 2: trimodal best/worst on Bolivia
    # ----------------------------------------------------------------------
    print("\n=== Figure 2: trimodal best/worst on Bolivia ===")
    tri_ckpt = Path(args.ckpt_dir) / "trimodal_unet_best.pt"
    bolivia_json = Path(args.logs_dir) / "trimodal_bolivia_results.json"
    if not tri_ckpt.exists() or not bolivia_json.exists():
        print(f"  [skip] need {tri_ckpt} and {bolivia_json}")
    else:
        best, worst = best_and_worst_chips(bolivia_json, k=3)
        ds = trimodal_dataset(args.data_root, args.splits_dir, "bolivia")
        model = load_trimodal(tri_ckpt, device)
        def pred_fn(sample):
            s1, s2, dem, _ = sample
            return predict_trimodal(model, s1, s2, dem, device)
        render_gallery(
            "TriModal U-Net on Bolivia (cross-region) — Best 3 (top) vs Worst 3 (bottom)",
            best + worst, ds, pred_fn,
            Path(args.fig_dir) / "paper_trimodal_bolivia_predictions.png",
        )
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Figure 3: cross-model disagreement on Bolivia_103757 (slide 11 example)
    # ----------------------------------------------------------------------
    print("\n=== Figure 3: Bolivia_103757 cross-model disagreement ===")
    needed = [
        Path(args.ckpt_dir) / "fcn_baseline_best.pt",
        Path(args.ckpt_dir) / "fusion_unet_best.pt",
        Path(args.ckpt_dir) / "trimodal_unet_best.pt",
    ]
    if not all(p.exists() for p in needed):
        print(f"  [skip] need all 3 primary checkpoints in {args.ckpt_dir}")
    else:
        render_disagreement(args, device,
                             Path(args.fig_dir) / "paper_disagreement_bolivia_103757.png")

    print("\nDone. rsync results/figures/paper_*.png back to local for the slides.")


if __name__ == "__main__":
    main()
