"""Per-chip evaluation: produce a JSON with metrics for every chip in a split.

Aggregate metrics (mean IoU, mean Dice across all pixels) hide per-chip
distributions. Many of the paper's analyses — bootstrapped CIs, paired
significance tests, error analysis, hard-chip identification — need
per-chip arrays.

Why a separate script: patching every training script to emit per-chip
JSONs would touch a lot of code and create version skew between the
training scripts. A dedicated post-hoc script that loads any checkpoint
and walks the split is cheaper and produces consistent JSONs across all
model families.

Output schema (JSON):
    {
      "model":       "trimodal" | "fusion" | "bimodal" | "ablation" | "fcn",
      "checkpoint":  "...",
      "split":       "test" | "val" | "bolivia",
      "modalities":  "s1_s2_dem" | ...                # ablation/bimodal only
      "aggregate":   { iou, dice, precision, recall, accuracy, tp, fp, tn, fn },
      "per_chip":    [
          { "chip": "<filename>", "iou", "dice", "precision", "recall",
            "accuracy", "tp", "fp", "tn", "fn", "water_fraction" },
          ...
      ]
    }

Usage:
    python scripts/eval_per_chip.py \\
        --model trimodal \\
        --checkpoint results/checkpoints/trimodal_p010_best.pt \\
        --data_root  ... --splits_dir ... --split test

    python scripts/eval_per_chip.py \\
        --model bimodal --modalities s1_dem \\
        --checkpoint results/checkpoints/bimodal_s1_dem_seed042_best.pt \\
        ...
"""

import argparse, json, sys
from pathlib import Path

import torch
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import compute_metrics


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------

def build_model(model_kind, modalities=None):
    """Return a model instance ready for evaluation."""
    if model_kind == "trimodal":
        from src.models.trimodal_unet import TriModalFusionUNet
        return TriModalFusionUNet()
    if model_kind == "fusion":
        from src.models.fusion_unet import FusionUNet
        return FusionUNet()
    if model_kind == "bimodal":
        from src.models.bimodal_cross_attn_unet import build_bimodal
        if modalities is None:
            raise ValueError("--modalities is required for --model bimodal")
        return build_bimodal(tuple(modalities.split("_")))
    if model_kind == "ablation":
        from src.models.early_fusion_unet import EarlyFusionUNet
        if modalities is None:
            raise ValueError("--modalities is required for --model ablation")
        ch = {"s1": 2, "s2": 13, "dem": 2}
        in_ch = sum(ch[m] for m in modalities.split("_"))
        return EarlyFusionUNet(in_channels=in_ch)
    if model_kind == "fcn":
        from src.models.fcn_baseline import FCNBaseline
        return FCNBaseline()
    raise ValueError(f"Unknown model kind: {model_kind}")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def get_loader(split, data_root, splits_dir):
    """Return the trimodal loader for the requested split (always batch=1)."""
    from src.data.dataset import get_trimodal_dataloaders
    loaders = get_trimodal_dataloaders(
        data_root=data_root, splits_dir=splits_dir,
        batch_size=1, num_workers=4, crop_size=None,
    )
    if split not in loaders:
        raise ValueError(f"Split {split} not in loaders {list(loaders)}")
    return loaders[split], loaders[split].dataset


def get_chip_filenames(dataset):
    """Pull the source filename for each sample in the dataset."""
    # Sen1Floods11TriModal stores `samples = [(s1_file, label_file), ...]`
    return [s1_file for s1_file, _ in dataset.samples]


# --------------------------------------------------------------------------
# Inference dispatch — model-specific input plumbing
# --------------------------------------------------------------------------

def run_model(model, model_kind, modalities, s1, s2, dem, device, use_amp):
    """Apply the model with the input signature appropriate for its kind.

    Dispatch table:
        fcn       -> model(s1)
        fusion    -> model(s1, s2)
        trimodal  -> model(s1, s2, dem)
        bimodal   -> model(a, b) in modalities order
        ablation  -> model(cat(modalities))    (early fusion concatenates)
    """
    s1, s2, dem = s1.to(device), s2.to(device), dem.to(device)
    pool = {"s1": s1, "s2": s2, "dem": dem}

    with autocast(device_type="cuda", enabled=use_amp):
        if model_kind == "fcn":
            out = model(s1)
        elif model_kind == "fusion":
            out = model(s1, s2)
        elif model_kind == "trimodal":
            out = model(s1, s2, dem)
        elif model_kind == "bimodal":
            keys = modalities.split("_")
            if len(keys) != 2:
                raise ValueError(f"bimodal expects 2 modalities, got {modalities}")
            out = model(pool[keys[0]], pool[keys[1]])
        elif model_kind == "ablation":
            keys = modalities.split("_")
            x = torch.cat([pool[k] for k in keys], dim=1)
            out = model(x)
        else:
            raise ValueError(f"Unknown model kind: {model_kind}")
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=["trimodal", "fusion", "bimodal", "ablation", "fcn"])
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--modalities",  default=None,
                   help="Required for --model bimodal/ablation; underscore-joined")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--output",      default=None,
                   help="Output JSON; default derived from checkpoint name")
    p.add_argument("--device",      default="auto")
    p.add_argument("--no_amp",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    # Build model + load checkpoint
    print(f"Building {args.model} (modalities={args.modalities})")
    model = build_model(args.model, args.modalities).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build data
    print(f"Loading split: {args.split}")
    loader, dataset = get_loader(args.split, args.data_root, args.splits_dir)
    chip_names = get_chip_filenames(dataset)
    print(f"  {len(dataset)} chips")

    # Run per-chip inference
    per_chip = []
    agg_tp = agg_fp = agg_tn = agg_fn = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="  Per-chip")):
            s1, s2, dem, label = batch
            label = label.to(device)
            out = run_model(model, args.model, args.modalities,
                            s1, s2, dem, device, use_amp)
            pred = torch.argmax(out, dim=1)
            m = compute_metrics(pred.cpu().numpy(), label.cpu().numpy(),
                                ignore_index=255)
            valid = (label.cpu().numpy() != 255).sum()
            water = ((label.cpu().numpy() == 1) & (label.cpu().numpy() != 255)).sum()
            water_fraction = float(water) / float(valid + 1e-7)

            per_chip.append({
                "chip":      chip_names[idx],
                "iou":       float(m["iou"]),
                "dice":      float(m["dice"]),
                "precision": float(m["precision"]),
                "recall":    float(m["recall"]),
                "accuracy":  float(m["accuracy"]),
                "tp":        int(m["tp"]),
                "fp":        int(m["fp"]),
                "tn":        int(m["tn"]),
                "fn":        int(m["fn"]),
                "water_fraction": water_fraction,
            })
            agg_tp += m["tp"]; agg_fp += m["fp"]
            agg_tn += m["tn"]; agg_fn += m["fn"]

    aggregate = {
        "tp": agg_tp, "fp": agg_fp, "tn": agg_tn, "fn": agg_fn,
        "iou":       agg_tp / (agg_tp + agg_fp + agg_fn + 1e-7),
        "dice":      2 * agg_tp / (2 * agg_tp + agg_fp + agg_fn + 1e-7),
        "precision": agg_tp / (agg_tp + agg_fp + 1e-7),
        "recall":    agg_tp / (agg_tp + agg_fn + 1e-7),
        "accuracy":  (agg_tp + agg_tn) / (agg_tp + agg_fp + agg_tn + agg_fn + 1e-7),
    }

    # Determine output path if not supplied
    if args.output is None:
        ckpt_stem = Path(args.checkpoint).stem.replace("_best", "")
        args.output = f"results/logs/{ckpt_stem}_{args.split}_per_chip.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":      args.model,
        "checkpoint": str(args.checkpoint),
        "split":      args.split,
        "modalities": args.modalities,
        "aggregate":  aggregate,
        "per_chip":   per_chip,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nAggregate IoU: {aggregate['iou']:.4f}  Dice: {aggregate['dice']:.4f}")
    print(f"Saved per-chip data ({len(per_chip)} chips) -> {args.output}")


if __name__ == "__main__":
    main()
