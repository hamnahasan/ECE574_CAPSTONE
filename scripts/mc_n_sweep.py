"""Sweep the number of MC-Dropout forward passes (N) on a trained checkpoint.

We currently report ECE at N=20 and *claim* it is the sweet spot. This
script tests that claim — runs MC-Dropout at N in {5, 10, 20, 50, 100}
on the same checkpoint and reports ECE, mean predictive variance, and
NLL at each N. The expected result is that ECE plateaus by N=20 and
gains beyond that are negligible. If it doesn't plateau, we update the
paper to use whichever N actually converges.

Outputs:
    results/logs/<run>_mc_n_sweep.json    machine-readable
    results/figures/<run>_mc_n_sweep.png  ECE-vs-N curve
    results/logs/<run>_mc_n_sweep.md      markdown table for the paper

Usage:
    python scripts/mc_n_sweep.py \\
        --model trimodal \\
        --checkpoint results/checkpoints/trimodal_unet_best.pt \\
        --data_root  ... --splits_dir ... --split test \\
        --n_values 5 10 20 50 100
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.uncertainty import mc_predict, compute_ece


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=["trimodal", "fusion", "bimodal"])
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--modalities",  default=None,
                   help="Required for --model bimodal (e.g. s1_dem)")
    p.add_argument("--data_root",   required=True)
    p.add_argument("--splits_dir",  required=True)
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--n_values",    nargs="+", type=int,
                   default=[5, 10, 20, 50, 100])
    p.add_argument("--n_bins",      type=int, default=15)
    p.add_argument("--seed",        type=int, default=42,
                   help="Seed only affects torch.manual_seed before MC sampling")
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--run_name",    default=None,
                   help="Override the run name used for output files")
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
        if modalities is None:
            raise ValueError("--modalities is required for --model bimodal")
        return build_bimodal(tuple(modalities.split("_")))
    raise ValueError(f"Unknown model: {model_kind}")


def collect_predictions(model, dataset, device, n_samples, model_kind, modalities):
    """Run MC at a single N. Returns concatenated (probs, labels) over the split."""
    all_probs  = []
    all_labels = []
    chip_uncs  = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        s1, s2, dem, label = sample  # always 4-tuple from trimodal dataset
        s1, s2, dem = s1.unsqueeze(0), s2.unsqueeze(0), dem.unsqueeze(0)
        # Run MC according to the model's input signature
        if model_kind == "trimodal":
            mean_prob, unc = mc_predict(model, s1, s2, dem, n_samples=n_samples,
                                         device=device)
        elif model_kind == "fusion":
            mean_prob, unc = mc_predict(model, s1, s2, n_samples=n_samples,
                                         device=device)
        elif model_kind == "bimodal":
            # mc_predict's signature was written for trimodal/fusion. For
            # bimodal we run our own loop.
            mean_prob, unc = bimodal_mc_predict(model, s1, s2, dem,
                                                 modalities, n_samples, device)
        else:
            raise ValueError(model_kind)

        lbl = label.numpy()
        valid = lbl != 255
        all_probs.append(mean_prob[valid])
        all_labels.append(lbl[valid])
        if valid.sum() > 0:
            chip_uncs.append(float(unc[valid].mean()))

    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            float(np.mean(chip_uncs)) if chip_uncs else 0.0)


@torch.no_grad()
def bimodal_mc_predict(model, s1, s2, dem, modalities, n_samples, device):
    """MC inference for the bimodal model (mc_predict doesn't know about it)."""
    from src.utils.uncertainty import enable_dropout
    import torch.nn.functional as TF

    pool = {"s1": s1.to(device), "s2": s2.to(device), "dem": dem.to(device)}
    a_key, b_key = modalities.split("_")
    a, b = pool[a_key], pool[b_key]

    model.eval()
    enable_dropout(model)
    probs = []
    for _ in range(n_samples):
        logits = model(a, b)
        probs.append(TF.softmax(logits, dim=1)[0, 1].cpu().float())
    stack = torch.stack(probs, dim=0)
    return stack.mean(0).numpy(), stack.var(0).numpy()


def nll_from_probs(probs, labels, eps=1e-7):
    """Negative log-likelihood for binary labels under predicted probs."""
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def brier_from_probs(probs, labels):
    """Brier score (mean squared error of probability vs label)."""
    return float(np.mean((probs - labels) ** 2))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset
    from src.data.dataset import get_trimodal_dataloaders
    loaders = get_trimodal_dataloaders(
        data_root=args.data_root, splits_dir=args.splits_dir,
        batch_size=1, num_workers=0, crop_size=None,
    )
    dataset = loaders[args.split].dataset
    print(f"Dataset: {args.split} ({len(dataset)} chips)")

    # Build & load model
    model = build_model(args.model, args.modalities).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {args.checkpoint}")

    # Sweep N values
    rows = []
    for n in sorted(args.n_values):
        print(f"\n=== N = {n} ===")
        t0 = time.time()
        probs, labels, mean_unc = collect_predictions(
            model, dataset, device, n, args.model, args.modalities,
        )
        elapsed = time.time() - t0
        ece, _ = compute_ece(probs, labels, n_bins=args.n_bins)
        nll    = nll_from_probs(probs, labels)
        brier  = brier_from_probs(probs, labels)
        print(f"  ECE={ece:.4f}  NLL={nll:.4f}  Brier={brier:.4f}  "
              f"meanVar={mean_unc:.5f}  time={elapsed:.1f}s")
        rows.append({
            "n_samples": n, "ece": ece, "nll": nll, "brier": brier,
            "mean_uncertainty": mean_unc, "wall_time_sec": elapsed,
        })

    # Output naming
    run_name = args.run_name or Path(args.checkpoint).stem.replace("_best", "")
    log_dir  = Path(args.output_dir) / "logs"
    fig_dir  = Path(args.output_dir) / "figures"
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    out_json = log_dir / f"{run_name}_mc_n_sweep.json"
    with open(out_json, "w") as f:
        json.dump({"run_name": run_name, "split": args.split,
                   "checkpoint": str(args.checkpoint),
                   "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Markdown table
    out_md = log_dir / f"{run_name}_mc_n_sweep.md"
    with open(out_md, "w") as f:
        f.write("| N | ECE | NLL | Brier | Mean uncertainty | Time (s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['n_samples']} | {r['ece']:.4f} | {r['nll']:.4f} | "
                    f"{r['brier']:.4f} | {r['mean_uncertainty']:.5f} | "
                    f"{r['wall_time_sec']:.1f} |\n")
    print(f"Saved: {out_md}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        ns   = [r["n_samples"] for r in rows]
        eces = [r["ece"]       for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ns, eces, "o-", color="#3182bd", linewidth=2, markersize=8)
        ax.axhline(0.05, linestyle="--", color="#666", linewidth=1,
                   label="0.05 well-calibrated threshold")
        ax.set_xscale("log")
        ax.set_xlabel("MC samples N (log scale)", fontsize=11)
        ax.set_ylabel("ECE (lower is better)", fontsize=11)
        ax.set_title(f"MC-Dropout N sweep — {run_name} ({args.split})", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        out_png = fig_dir / f"{run_name}_mc_n_sweep.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")


if __name__ == "__main__":
    main()
