"""Generate train/val CSVs for the weakly-labeled Sen1Floods11 chips.

The official v1.1 release ships ~4 385 weakly-labeled S1+S2 chips under
``data/flood_events/WeaklyLabeled/`` but does NOT ship a corresponding
splits CSV under ``splits/flood_weakly_labeled/``. This script walks the
S1Weak directory, pairs each chip with its weak-label counterpart, and
writes two-column CSVs in the same format as the hand-labeled splits so
the existing dataloaders work without changes.

Bonafilia 2020 ships TWO weak-label sources:

  S1OtsuLabelWeak/     <- Otsu threshold on S1 VH (default; cloud-immune)
  S2IndexLabelWeak/    <- spectral index threshold on S2 (cloud-sensitive)

Default is S1OtsuLabelWeak because the paper uses it as the primary weak
label source for SAR+optical training; pass --label_token S2IndexLabelWeak
to use the S2-derived labels instead.

Usage:
    python scripts/make_weak_splits.py \\
        --s1_dir     /lustre/.../WeaklyLabeled/S1Weak \\
        --output_dir /lustre/.../v1.1/splits/flood_weakly_labeled \\
        --val_fraction 0.05 --seed 42

Outputs:
    {output_dir}/flood_train_data.csv
    {output_dir}/flood_valid_data.csv
"""

import argparse
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s1_dir",     required=True,
                   help="Path to S1Weak directory")
    p.add_argument("--output_dir", required=True,
                   help="Where to write flood_train/valid_data.csv")
    p.add_argument("--label_token", default="S1OtsuLabelWeak",
                   choices=["S1OtsuLabelWeak", "S2IndexLabelWeak"],
                   help="Which weak-label source to pair the S1 chips with")
    p.add_argument("--val_fraction", type=float, default=0.05,
                   help="Fraction held out as validation (deterministic by seed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    s1_dir = Path(args.s1_dir)
    if not s1_dir.is_dir():
        raise SystemExit(f"Not a directory: {s1_dir}")

    # Filter out macOS metadata files (._<name>.tif) that ship alongside the
    # real chips on Lustre. Path.glob() doesn't honor the shell hidden-file
    # convention, so we filter explicitly.
    files = sorted(
        p.name for p in s1_dir.glob("*.tif")
        if not p.name.startswith("._")
    )
    if not files:
        raise SystemExit(f"No .tif files found in {s1_dir}")
    print(f"Found {len(files)} S1Weak chips (after filtering ._ junk)")

    # Sanity check — the label folder should exist alongside S1Weak
    label_dir = s1_dir.parent / args.label_token
    if not label_dir.is_dir():
        print(f"[WARN] Label dir not found at {label_dir}. The split CSV "
              "will still be generated but training will fail until labels "
              "are present.")

    rng = random.Random(args.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * args.val_fraction)))
    val_files   = shuffled[:n_val]
    train_files = shuffled[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(rows, path):
        with open(path, "w") as f:
            for s1 in rows:
                label = s1.replace("S1Weak", args.label_token)
                f.write(f"{s1},{label}\n")

    train_path = out_dir / "flood_train_data.csv"
    val_path   = out_dir / "flood_valid_data.csv"
    write_csv(train_files, train_path)
    write_csv(val_files,   val_path)

    print(f"Train: {len(train_files):4d}  -> {train_path}")
    print(f"Val:   {len(val_files):4d}  -> {val_path}")
    print(f"Label source: {args.label_token}")


if __name__ == "__main__":
    main()
