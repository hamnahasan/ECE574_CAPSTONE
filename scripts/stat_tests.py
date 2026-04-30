"""Statistical tests on per-chip metrics.

Consumes the per-chip JSONs produced by `scripts/eval_per_chip.py` and
runs three tests / summaries that the paper needs:

  1. **Bootstrap confidence intervals.** Resample chips with replacement
     B times (default 10 000), recompute the mean of each metric on each
     bootstrap sample, return the 2.5 / 97.5 percentiles as the 95% CI.
     Reported in the main results table alongside the point estimate.

  2. **Paired t-test.** For two models on the same split, test whether
     the mean per-chip IoU difference is significantly different from zero.
     Standard parametric test; assumes approximate normality.

  3. **Wilcoxon signed-rank.** Non-parametric counterpart to the t-test;
     reported alongside it because reviewers will ask. Robust to outliers
     and per-chip distributions that aren't gaussian.

Inputs are JSONs produced by eval_per_chip.py. Each must contain a
`per_chip` array of dicts with at least `chip` and `iou` fields. The script
matches per-chip records by filename across the two models so the tests
are paired.

Usage examples:

    # Bootstrap CI for one model
    python scripts/stat_tests.py ci \\
        --json results/logs/trimodal_unet_test_per_chip.json --metric iou

    # Paired tests between two models
    python scripts/stat_tests.py paired \\
        --json_a results/logs/trimodal_unet_test_per_chip.json \\
        --json_b results/logs/ablation_s1_s2_dem_test_per_chip.json \\
        --metric iou
"""

import argparse, json
from pathlib import Path

import numpy as np
from scipy import stats as sstats


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def load_per_chip(json_path):
    with open(json_path) as f:
        d = json.load(f)
    if "per_chip" not in d:
        raise ValueError(f"{json_path} has no 'per_chip' array — was it "
                         "produced by scripts/eval_per_chip.py?")
    return d["per_chip"]


def array(per_chip, metric):
    """Pull the metric column as a numpy array."""
    return np.array([float(c[metric]) for c in per_chip], dtype=np.float64)


def aligned_arrays(pc_a, pc_b, metric):
    """Inner-join two per-chip lists by filename, return matched arrays."""
    by_a = {c["chip"]: float(c[metric]) for c in pc_a}
    by_b = {c["chip"]: float(c[metric]) for c in pc_b}
    common = sorted(set(by_a) & set(by_b))
    a = np.array([by_a[k] for k in common])
    b = np.array([by_b[k] for k in common])
    return common, a, b


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def bootstrap_ci(values, n_resamples=10_000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return {
        "mean":   float(values.mean()),
        "median": float(np.median(values)),
        "ci_low":  float(lo),
        "ci_high": float(hi),
        "n_chips": int(n),
        "n_resamples": int(n_resamples),
        "alpha":   float(alpha),
    }


def paired_tests(a, b, alpha=0.05):
    """Paired t-test + Wilcoxon signed-rank on (a - b).

    Positive diff means a > b on that chip. Both two-sided.
    """
    diff = a - b
    t_stat, t_p = sstats.ttest_rel(a, b)
    try:
        w_stat, w_p = sstats.wilcoxon(diff, zero_method="wilcox",
                                       alternative="two-sided")
    except ValueError:
        # All-zero differences -> wilcoxon raises
        w_stat, w_p = 0.0, 1.0
    return {
        "n_pairs":         int(len(a)),
        "mean_diff":       float(diff.mean()),
        "median_diff":     float(np.median(diff)),
        "ttest_statistic": float(t_stat),
        "ttest_p":         float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p":      float(w_p),
        "significant_at": {
            "alpha_5pct":  bool((t_p < 0.05) and (w_p < 0.05)),
            "alpha_1pct":  bool((t_p < 0.01) and (w_p < 0.01)),
        },
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_ci(args):
    per_chip = load_per_chip(args.json)
    vals     = array(per_chip, args.metric)
    res = bootstrap_ci(vals, n_resamples=args.n_resamples,
                        alpha=args.alpha, seed=args.seed)
    out = {"json": str(args.json), "metric": args.metric, "result": res}
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output}")


def cmd_paired(args):
    pc_a = load_per_chip(args.json_a)
    pc_b = load_per_chip(args.json_b)
    common, a, b = aligned_arrays(pc_a, pc_b, args.metric)
    if len(common) < 5:
        print(f"WARNING: only {len(common)} chips matched between the two JSONs")
    res = paired_tests(a, b, alpha=args.alpha)
    res["a_mean"] = float(a.mean())
    res["b_mean"] = float(b.mean())
    out = {
        "json_a": str(args.json_a), "json_b": str(args.json_b),
        "metric": args.metric, "result": res,
    }
    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("ci", help="Bootstrap CI for one model")
    pc.add_argument("--json", required=True)
    pc.add_argument("--metric", default="iou")
    pc.add_argument("--n_resamples", type=int, default=10_000)
    pc.add_argument("--alpha", type=float, default=0.05)
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--output", default=None)
    pc.set_defaults(func=cmd_ci)

    pp = sub.add_parser("paired", help="Paired t-test + Wilcoxon between two models")
    pp.add_argument("--json_a", required=True)
    pp.add_argument("--json_b", required=True)
    pp.add_argument("--metric", default="iou")
    pp.add_argument("--alpha", type=float, default=0.05)
    pp.add_argument("--output", default=None)
    pp.set_defaults(func=cmd_paired)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
