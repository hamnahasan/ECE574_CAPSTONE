"""Deep-ensemble inference: average predictions from N independently-trained
models (different seeds) and report aggregate metrics + ECE.

Standard alternative to MC Dropout for uncertainty quantification. The
expected story for the paper is one of:

    a) Deep ensemble matches or slightly beats MC Dropout on calibration
       (typical literature finding) but at 5x training cost — we report
       both and recommend MC Dropout as the practical default.

    b) MC Dropout is sufficient and the ensemble doesn't help — we report
       both and lean harder on MC Dropout.

Either way reviewers expect to see this comparison.

Usage:
    python scripts/deep_ensemble.py \\
        --model trimodal \\
        --checkpoints results/checkpoints/trimodal_p010_seed042_best.pt \\
                      results/checkpoints/trimodal_p010_seed123_best.pt \\
                      results/checkpoints/trimodal_p010_seed007_best.pt \\
        --data_root  ... --splits_dir ... --split test
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import compute_metrics
from src.utils.uncertainty import compute_ece


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=["trimodal", "fusion", "bimodal"])
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Two or more best.pt checkpoints to ensemble")
    p.add_argument("--modalities",  default=None,
                   help="Required for --model bimodal")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--n_bins",      type=int, default=15)
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


def model_forward(model, model_kind, modalities, s1, s2, dem, device, use_amp):
    pool = {"s1": s1.to(device), "s2": s2.to(device), "dem": dem.to(device)}
    with autocast(device_type="cuda", enabled=use_amp):
        if model_kind == "trimodal":
            return model(pool["s1"], pool["s2"], pool["dem"])
        if model_kind == "fusion":
            return model(pool["s1"], pool["s2"])
        if model_kind == "bimodal":
            a, b = modalities.split("_")
            return model(pool[a], pool[b])
    raise ValueError(model_kind)


def main():
    args = parse_args()
    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models = len(args.checkpoints)
    if n_models < 2:
        raise ValueError("Deep ensemble requires at least 2 checkpoints")
    print(f"Ensembling {n_models} checkpoints on {device}")

    # Load each checkpoint into a separate model instance. Memory permitting,
    # we hold all in GPU RAM and run them in a loop per chip; alternative is
    # to round-robin them through one slot.
    models = []
    for ckpt_path in args.checkpoints:
        m = build_model(args.model, args.modalities).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  Loaded {ckpt_path}")

    # Data
    from src.data.dataset import get_trimodal_dataloaders
    loaders = get_trimodal_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=1, num_workers=0, crop_size=None,
    )
    loader = loaders[args.split]
    print(f"Split: {args.split} ({len(loader)} chips)")

    # Inference
    all_probs  = []
    all_labels = []
    per_chip   = []
    agg_tp = agg_fp = agg_tn = agg_fn = 0
    with torch.no_grad():
        for s1, s2, dem, label in tqdm(loader, desc="  Ensemble"):
            label = label.to(device)
            # Average softmax probabilities across all ensemble members
            prob_sum = None
            for m in models:
                logits = model_forward(m, args.model, args.modalities,
                                        s1, s2, dem, device, use_amp)
                prob   = TF.softmax(logits, dim=1)
                prob_sum = prob if prob_sum is None else prob_sum + prob
            mean_prob = prob_sum / n_models                  # (1, 2, H, W)
            water_p   = mean_prob[0, 1].cpu().numpy()         # (H, W)
            pred      = torch.argmax(mean_prob, dim=1)        # (1, H, W)

            m_chip = compute_metrics(pred.cpu().numpy(), label.cpu().numpy(),
                                      ignore_index=255)
            agg_tp += m_chip["tp"]; agg_fp += m_chip["fp"]
            agg_tn += m_chip["tn"]; agg_fn += m_chip["fn"]
            per_chip.append({k: float(m_chip[k]) for k in
                             ["iou", "dice", "precision", "recall", "accuracy"]})

            lbl_np = label.cpu().numpy().squeeze()
            valid  = lbl_np != 255
            all_probs.append(water_p[valid])
            all_labels.append(lbl_np[valid])

    # Aggregate metrics from accumulated TP/FP/TN/FN
    aggregate = {
        "iou":       agg_tp / (agg_tp + agg_fp + agg_fn + 1e-7),
        "dice":      2 * agg_tp / (2 * agg_tp + agg_fp + agg_fn + 1e-7),
        "precision": agg_tp / (agg_tp + agg_fp + 1e-7),
        "recall":    agg_tp / (agg_tp + agg_fn + 1e-7),
        "accuracy":  (agg_tp + agg_tn) / (agg_tp + agg_fp + agg_tn + agg_fn + 1e-7),
        "tp": agg_tp, "fp": agg_fp, "tn": agg_tn, "fn": agg_fn,
    }
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    ece, _ = compute_ece(probs, labels, n_bins=args.n_bins)

    print(f"\nEnsemble of {n_models} models on {args.split}:")
    print(f"  IoU={aggregate['iou']:.4f}  Dice={aggregate['dice']:.4f}  ECE={ece:.4f}")

    run_name = args.run_name or f"ensemble_{args.model}_{n_models}models"
    log_dir  = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{run_name}_{args.split}_ensemble.json"
    with open(out, "w") as f:
        json.dump({
            "model": args.model, "checkpoints": list(args.checkpoints),
            "split": args.split, "n_models": n_models,
            "aggregate": aggregate, "ece": ece,
            "per_chip": per_chip,
        }, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
