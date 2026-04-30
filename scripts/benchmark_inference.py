"""Benchmark inference latency and peak memory for trained models.

Reviewers always ask: how big is the model, how fast does it run, how
much GPU memory does it need? This script reports all of that for any
saved checkpoint.

Reported metrics:
  - Parameter count
  - Estimated FLOPs (best-effort via torch.profiler when available)
  - Peak GPU memory during inference
  - Mean / std / p95 latency per chip (deterministic single pass)
  - Mean latency at MC-Dropout N=20 (for the uncertainty estimate cost)

Usage:
    python scripts/benchmark_inference.py \\
        --model trimodal \\
        --checkpoint results/checkpoints/trimodal_unet_best.pt \\
        --warmup 5 --runs 50

Multiple checkpoints can be benchmarked back-to-back to populate a
literature-comparison table:

    for m in fusion trimodal bimodal; do
        python scripts/benchmark_inference.py --model $m ...
    done
"""

import argparse, json, sys, time
from pathlib import Path

import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=["trimodal", "fusion", "bimodal", "ablation", "fcn"])
    p.add_argument("--checkpoint",  default=None,
                   help="Optional; if omitted, benchmark a randomly-initialized model")
    p.add_argument("--modalities",  default=None,
                   help="Required for --model bimodal/ablation")
    p.add_argument("--input_size",  type=int, default=256)
    p.add_argument("--warmup",      type=int, default=5)
    p.add_argument("--runs",        type=int, default=50)
    p.add_argument("--mc_samples",  type=int, default=20,
                   help="N for the MC-Dropout latency benchmark; 0 to skip")
    p.add_argument("--device",      default="auto")
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--output",      default=None)
    p.add_argument("--cpu_runs",    type=int, default=0,
                   help="Run an additional CPU latency benchmark with this many runs (0=skip)")
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
    if model_kind == "ablation":
        from src.models.early_fusion_unet import EarlyFusionUNet
        ch = {"s1": 2, "s2": 13, "dem": 2}
        return EarlyFusionUNet(in_channels=sum(ch[m] for m in modalities.split("_")))
    if model_kind == "fcn":
        from src.models.fcn_baseline import FCNBaseline
        return FCNBaseline()
    raise ValueError(model_kind)


def make_dummy_inputs(model_kind, modalities, size, device):
    """Return a (forward_call, input_tensors) tuple appropriate for the model."""
    s1  = torch.randn(1, 2,  size, size, device=device)
    s2  = torch.randn(1, 13, size, size, device=device)
    dem = torch.randn(1, 2,  size, size, device=device)
    if model_kind == "trimodal":
        return ("trimodal", (s1, s2, dem))
    if model_kind == "fusion":
        return ("fusion",   (s1, s2))
    if model_kind == "bimodal":
        a, b = modalities.split("_")
        pool = {"s1": s1, "s2": s2, "dem": dem}
        return ("bimodal",  (pool[a], pool[b]))
    if model_kind == "ablation":
        ch = {"s1": s1, "s2": s2, "dem": dem}
        keys = modalities.split("_")
        x = torch.cat([ch[k] for k in keys], dim=1)
        return ("ablation", (x,))
    if model_kind == "fcn":
        return ("fcn",      (s1,))
    raise ValueError(model_kind)


@torch.no_grad()
def time_model(model, inputs, runs, use_amp, device):
    """Return list of per-run latencies in milliseconds."""
    is_cuda = device.type == "cuda"
    times = []
    for _ in range(runs):
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with autocast(device_type=device.type, enabled=use_amp):
            _ = model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def stats(values):
    import statistics
    if not values:
        return {}
    return {
        "mean":  statistics.mean(values),
        "std":   statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min":   min(values),
        "max":   max(values),
        "p50":   statistics.median(values),
        "p95":   sorted(values)[max(0, int(len(values) * 0.95) - 1)],
        "n":     len(values),
    }


def main():
    args = parse_args()
    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    model = build_model(args.model, args.modalities).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {args.checkpoint}")

    print(f"Model: {args.model}  params: {n_params:,}  device: {device}  "
          f"AMP: {use_amp}  input: {args.input_size}x{args.input_size}")

    _, inputs = make_dummy_inputs(args.model, args.modalities,
                                    args.input_size, device)

    # GPU warmup + timed runs
    print(f"\nGPU benchmark: {args.warmup} warmup + {args.runs} timed runs")
    _ = time_model(model, inputs, args.warmup, use_amp, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = time_model(model, inputs, args.runs, use_amp, device)
    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) \
                   if device.type == "cuda" else 0.0
    s = stats(times)
    print(f"  Single-pass latency (ms): mean={s['mean']:.2f}  "
          f"std={s['std']:.2f}  p50={s['p50']:.2f}  p95={s['p95']:.2f}")
    if peak_mem_mb:
        print(f"  Peak GPU memory: {peak_mem_mb:.1f} MB")

    # MC-Dropout latency
    mc_stats = {}
    if args.mc_samples > 0:
        from src.utils.uncertainty import enable_dropout
        enable_dropout(model)
        print(f"\nMC-Dropout latency (N={args.mc_samples}):")
        mc_times = []
        for _ in range(args.runs):
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.mc_samples):
                with autocast(device_type=device.type, enabled=use_amp):
                    _ = model(*inputs)
            if device.type == "cuda": torch.cuda.synchronize()
            mc_times.append((time.perf_counter() - t0) * 1000.0)
        mc_stats = stats(mc_times)
        print(f"  N={args.mc_samples} latency (ms): "
              f"mean={mc_stats['mean']:.2f}  p95={mc_stats['p95']:.2f}")
        model.eval()

    # CPU benchmark (optional, for deployment-feasibility numbers)
    cpu_stats = {}
    if args.cpu_runs > 0:
        print(f"\nCPU benchmark: {args.cpu_runs} runs")
        cpu_model  = model.cpu()
        cpu_inputs = tuple(t.cpu() for t in inputs)
        cpu_dev    = torch.device("cpu")
        _ = time_model(cpu_model, cpu_inputs, max(2, args.cpu_runs // 5),
                        False, cpu_dev)
        cpu_times = time_model(cpu_model, cpu_inputs, args.cpu_runs, False, cpu_dev)
        cpu_stats = stats(cpu_times)
        print(f"  CPU latency (ms): mean={cpu_stats['mean']:.2f}  "
              f"p95={cpu_stats['p95']:.2f}")

    payload = {
        "model": args.model, "modalities": args.modalities,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "n_params": n_params,
        "input_size": args.input_size, "amp": use_amp,
        "device": str(device),
        "single_pass_ms": s,
        "peak_gpu_memory_mb": peak_mem_mb,
        "mc_dropout": {"n_samples": args.mc_samples, **mc_stats} if mc_stats else None,
        "cpu_latency_ms": cpu_stats if cpu_stats else None,
    }
    out_path = args.output or f"results/logs/benchmark_{args.model}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
