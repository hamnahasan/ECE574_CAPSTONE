"""Aggregate per-seed test-result JSONs into mean +/- std summaries.

Inputs are the JSON files produced by the various training scripts when
they finish, e.g.:

    results/logs/trimodal_p010_seed042_test_results.json
    results/logs/trimodal_p010_seed123_test_results.json
    results/logs/trimodal_p010_seed007_test_results.json

This script discovers all seed-tagged variants of one base run name,
loads them, and writes:

    results/logs/<run_name>_seeds_summary.json  (machine-readable)

Plus a markdown table to stdout that you can paste into the paper draft.

Usage:
    python scripts/aggregate_seeds.py --run_name trimodal_p010
    python scripts/aggregate_seeds.py --run_name bimodal_s1_dem
    python scripts/aggregate_seeds.py --run_name trimodal_p010 \\
        --extra_runs trimodal_p020 trimodal_p030

The --extra_runs option lets you build a single comparison table across
multiple base names (e.g., the whole modality-dropout sweep).
"""

import argparse, json, re
from pathlib import Path
from statistics import mean, pstdev


LOGS_DIR = Path("results/logs")

SEED_PATTERN = re.compile(r"_seed(\d{3})_test_results\.json$")
METRICS = ["iou", "dice", "precision", "recall", "accuracy"]


def discover_seed_files(run_name):
    """Find every per-seed JSON for a given base run name.

    Returns a dict: {seed_int: path_to_json}.
    """
    found = {}
    pattern = f"{run_name}_seed*_test_results.json"
    for path in LOGS_DIR.glob(pattern):
        m = SEED_PATTERN.search(path.name)
        if m:
            found[int(m.group(1))] = path
    return found


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def aggregate_one_run(run_name):
    """Build a single row: {metric: {mean, std, n}} for one base run name."""
    files = discover_seed_files(run_name)
    if not files:
        print(f"[WARN] No per-seed files for {run_name} (looked under {LOGS_DIR})")
        return None

    # Each file is a flat metrics dict at the top level
    arrays = {m: [] for m in METRICS}
    for seed, path in sorted(files.items()):
        m = load_metrics(path)
        for k in METRICS:
            if k in m:
                arrays[k].append(float(m[k]))

    summary = {}
    for k, vals in arrays.items():
        if vals:
            summary[k] = {
                "mean":  mean(vals),
                "std":   pstdev(vals) if len(vals) > 1 else 0.0,
                "n":     len(vals),
                "values": vals,
            }
    summary["seeds"]  = sorted(files.keys())
    summary["sources"] = {str(s): str(p) for s, p in sorted(files.items())}
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   required=True,
                   help="Base run name (without _seedNNN suffix)")
    p.add_argument("--extra_runs", nargs="*", default=[],
                   help="Additional base run names to include in the table")
    p.add_argument("--write_summary", action="store_true",
                   help="Write a JSON summary file alongside the per-seed JSONs")
    args = p.parse_args()

    runs = [args.run_name] + args.extra_runs
    summaries = {}
    for r in runs:
        s = aggregate_one_run(r)
        if s is not None:
            summaries[r] = s

    if not summaries:
        print("No data found.")
        return

    # Print markdown table
    print("\n| Run | n | IoU | Dice | Precision | Recall |")
    print("|---|---|---|---|---|---|")
    for r, s in summaries.items():
        n   = s.get("iou", {}).get("n", 0)
        iou = s.get("iou", {})
        dic = s.get("dice", {})
        pre = s.get("precision", {})
        rec = s.get("recall", {})
        def fmt(d):
            if not d: return "—"
            return f"{d['mean']:.4f} ± {d['std']:.4f}"
        print(f"| {r} | {n} | {fmt(iou)} | {fmt(dic)} | {fmt(pre)} | {fmt(rec)} |")

    if args.write_summary:
        out_path = LOGS_DIR / f"{args.run_name}_seeds_summary.json"
        with open(out_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
