"""Evaluate a TriModal U-Net under missing-modality scenarios at inference.

For each of seven scenarios — full input + 3 single-drop + 3 single-only —
load a trained checkpoint, run the model on the test set with the specified
modalities replaced by zero tensors, and record IoU/Dice/Precision/Recall.

This is the second half of the modality-dropout sensitivity story: the first
half (slurm/sensitivity_mod_dropout.sbatch) trains six TriModal models at
different `p` values; this script measures how well each one tolerates a
missing modality at test time. Together they answer "does training-time
modality dropout buy inference-time robustness, and at what cost?"

Usage (single run):
    python scripts/eval_modality_robustness.py \\
        --run_name trimodal_p010 \\
        --data_root  /lustre/.../HandLabeled \\
        --splits_dir /lustre/.../flood_handlabeled

Usage (sweep — call once per p value):
    for tag in 000 005 010 020 030 050; do
        python scripts/eval_modality_robustness.py --run_name trimodal_p${tag} ...
    done

Outputs `{output_dir}/logs/{run_name}_robustness.json` with shape:
    { "scenarios": { "all": {iou, dice, precision, recall, accuracy},
                     "no_s1": {...}, ..., "dem_only": {...} },
      "run_name": "trimodal_p010", "checkpoint": "..." }
"""

import argparse, json, sys
from pathlib import Path

import torch
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_trimodal_dataloaders
from src.models.trimodal_unet import TriModalFusionUNet
from src.utils.metrics import MetricAccumulator


# Each scenario is (s1_keep, s2_keep, dem_keep). True = use real input,
# False = replace with zeros at inference. "all" is the baseline.
SCENARIOS = {
    "all":      (True,  True,  True),
    "no_s1":    (False, True,  True),
    "no_s2":    (True,  False, True),
    "no_dem":   (True,  True,  False),
    "s1_only":  (True,  False, False),
    "s2_only":  (False, True,  False),
    "dem_only": (False, False, True),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   required=True,
                   help="Run name prefix; checkpoint is {output_dir}/checkpoints/{run_name}_best.pt")
    p.add_argument("--data_root",  required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--split",      default="test", choices=["val", "test"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",     default="auto")
    p.add_argument("--no_amp",     action="store_true")
    return p.parse_args()


def evaluate_scenario(model, loader, device, use_amp, keep_s1, keep_s2, keep_dem):
    """Run a full pass with the specified modalities zeroed when keep=False."""
    model.eval()
    acc = MetricAccumulator()
    desc = f"keep s1={int(keep_s1)} s2={int(keep_s2)} dem={int(keep_dem)}"
    with torch.no_grad():
        for s1, s2, dem, labels in tqdm(loader, desc=desc, leave=False):
            s1     = s1.to(device)
            s2     = s2.to(device)
            dem    = dem.to(device)
            labels = labels.to(device)

            if not keep_s1:  s1  = torch.zeros_like(s1)
            if not keep_s2:  s2  = torch.zeros_like(s2)
            if not keep_dem: dem = torch.zeros_like(dem)

            with autocast(device_type="cuda", enabled=use_amp):
                out = model(s1, s2, dem)
            acc.update(torch.argmax(out, dim=1), labels, ignore_index=255)
    return acc.compute()


def main():
    args = parse_args()
    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    log_dir  = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{args.run_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Device: {device}  |  AMP: {use_amp}")
    print(f"Loading: {ckpt_path}")

    model = TriModalFusionUNet().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Test dataloader only — we never want to evaluate scenarios on training data.
    loaders = get_trimodal_dataloaders(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        batch_size=1,
        num_workers=args.num_workers,
        crop_size=256,
    )
    loader = loaders[args.split]

    results = {}
    for name, (k1, k2, kd) in SCENARIOS.items():
        m = evaluate_scenario(model, loader, device, use_amp, k1, k2, kd)
        results[name] = m
        print(f"  {name:9s}  IoU={m['iou']:.4f}  Dice={m['dice']:.4f}  "
              f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")

    out_path = log_dir / f"{args.run_name}_robustness.json"
    with open(out_path, "w") as f:
        json.dump({
            "run_name":   args.run_name,
            "checkpoint": str(ckpt_path),
            "split":      args.split,
            "scenarios":  results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
