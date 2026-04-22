"""Run MC Dropout uncertainty evaluation on a trained model.

Produces:
  - ECE (Expected Calibration Error) on the test or Bolivia split
  - Reliability diagram (confidence vs accuracy)
  - Per-chip mean predictive uncertainty
  - Sample uncertainty maps saved as PNG

Usage:
    python scripts/mc_uncertainty.py \
        --checkpoint results/checkpoints/trimodal_unet_best.pt \
        --model trimodal \
        --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
        --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
        --split test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.uncertainty import evaluate_uncertainty, mc_predict, reliability_diagram
from src.utils.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="MC Dropout uncertainty evaluation")
    p.add_argument("--checkpoint", required=True, help="Trained model .pt")
    p.add_argument("--model", required=True, choices=["fusion", "trimodal"],
                   help="Which model architecture the checkpoint corresponds to")
    p.add_argument("--data_root", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test", "bolivia"])
    p.add_argument("--n_samples", type=int, default=20,
                   help="Number of MC Dropout forward passes")
    p.add_argument("--n_bins", type=int, default=15,
                   help="Calibration bins for ECE")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--n_save_maps", type=int, default=6,
                   help="Number of sample uncertainty maps to save")
    return p.parse_args()


def build_model(model_name, checkpoint_path, device):
    """Load the correct model architecture and its weights."""
    if model_name == "fusion":
        from src.models.fusion_unet import FusionUNet
        model = FusionUNet(s1_channels=2, s2_channels=13,
                           num_classes=2, dropout_rate=0.1)
    elif model_name == "trimodal":
        from src.models.trimodal_unet import TriModalFusionUNet
        model = TriModalFusionUNet(s1_channels=2, s2_channels=13,
                                   dem_channels=2, num_classes=2,
                                   dropout_rate=0.1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Loaded {model_name} from epoch {ckpt.get('epoch','?')} "
          f"(val_iou={ckpt.get('val_iou', '?')})")
    return model


def build_dataset(model_name, data_root, splits_dir, split):
    """Build the appropriate dataset for this model."""
    from src.data.dataset import Sen1Floods11MultiModal, Sen1Floods11TriModal

    split_map = {
        "train": "flood_train_data.csv",
        "val":   "flood_valid_data.csv",
        "test":  "flood_test_data.csv",
        "bolivia": "flood_bolivia_data.csv",
    }
    data_root  = Path(data_root); splits_dir = Path(splits_dir)
    csv        = splits_dir / split_map[split]

    if model_name == "fusion":
        return Sen1Floods11MultiModal(
            split_csv=csv,
            s1_dir=data_root / "S1Hand",
            s2_dir=data_root / "S2Hand",
            label_dir=data_root / "LabelHand",
            crop_size=None, augment=False, normalize=True,
        )
    else:  # trimodal
        return Sen1Floods11TriModal(
            split_csv=csv,
            s1_dir=data_root / "S1Hand",
            s2_dir=data_root / "S2Hand",
            dem_dir=data_root / "DEMHand",
            label_dir=data_root / "LabelHand",
            crop_size=None, augment=False, normalize=True,
        )


def save_uncertainty_maps(model, dataset, device, n_samples, n_maps, save_dir):
    """Save a small gallery of uncertainty maps to PNG.

    These visualizations let us confirm that high uncertainty coincides with
    visually ambiguous regions (flood boundaries, cloud shadows, etc.), which
    is the qualitative evidence that MC Dropout is capturing meaningful
    uncertainty rather than random noise.
    """
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pick chips evenly spaced across the dataset for a representative sample
    indices = np.linspace(0, len(dataset) - 1, n_maps).astype(int).tolist()

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        if len(sample) == 4:
            s1, s2, dem, label = sample
            mean_prob, unc = mc_predict(
                model, s1.unsqueeze(0), s2.unsqueeze(0), dem.unsqueeze(0),
                n_samples=n_samples, device=device,
            )
        else:
            s1, s2, label = sample
            mean_prob, unc = mc_predict(
                model, s1.unsqueeze(0), s2.unsqueeze(0),
                n_samples=n_samples, device=device,
            )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        lbl_disp = np.where(label.numpy() == 255, 0.5,
                             label.numpy().astype(float))
        axes[0].imshow(lbl_disp, cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title("Ground truth"); axes[0].axis("off")

        axes[1].imshow(mean_prob, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Mean water probability"); axes[1].axis("off")

        im2 = axes[2].imshow(unc, cmap="magma")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, label="variance")
        axes[2].set_title("Predictive uncertainty"); axes[2].axis("off")

        chip_name = Path(dataset.samples[idx][0]).stem.replace("_S1Hand","")
        fig.suptitle(f"MC Dropout — {chip_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / f"uncertainty_map_{i:02d}_{chip_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {n_maps} uncertainty maps to {save_dir}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model   = build_model(args.model, args.checkpoint, device)
    dataset = build_dataset(args.model, args.data_root, args.splits_dir, args.split)
    print(f"Dataset: {args.split} ({len(dataset)} chips)")

    # Full dataset evaluation — ECE + reliability diagram
    log_dir = Path(args.output_dir) / "logs"
    fig_dir = Path(args.output_dir) / "figures" / f"uncertainty_{args.model}_{args.split}"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_uncertainty(
        model, dataset, device,
        n_samples=args.n_samples,
        n_bins=args.n_bins,
        save_dir=str(fig_dir),
    )

    # Sample maps
    save_uncertainty_maps(model, dataset, device,
                          n_samples=args.n_samples,
                          n_maps=args.n_save_maps,
                          save_dir=fig_dir)

    # Save JSON summary
    summary = {
        "model":      args.model,
        "split":      args.split,
        "n_samples":  args.n_samples,
        "n_bins":     args.n_bins,
        "ece":        results["ece"],
        "mean_uncertainty": results["mean_uncertainty"],
        "bins":       results["bins"],
    }
    out_path = log_dir / f"{args.model}_{args.split}_uncertainty.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
