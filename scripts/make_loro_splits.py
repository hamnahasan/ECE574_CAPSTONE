"""Generate leave-one-region-out (LORO) splits for cross-region evaluation.

Sen1Floods11 chips are named with the source event/country as the prefix:
``Bolivia_103757_S1Hand.tif``, ``Ghana_473553_S1Hand.tif``, etc. The dataset
covers 11 events. The standard `flood_train/valid/test` split mixes events
across all three subsets, which is why our headline test IoU (~0.78) is
in-distribution.

Real disaster-response generalization requires holding out an entire event.
Bolivia held-out (which we already report) is one such test; this script
generalizes that to *every* event so we can produce a leave-one-region-out
table — train on 10 events, test on the 11th, repeat.

For each event, the script writes a new splits directory:

    {output_dir}/loro_{event}/
        flood_train_data.csv      # all chips NOT in this event, minus val
        flood_valid_data.csv      # 10% of non-event chips, deterministic by seed
        flood_test_data.csv       # all chips of this event

The CSV format matches the original two-column convention
(s1_filename, label_filename) so existing dataloaders work without changes.

Usage:
    python scripts/make_loro_splits.py \\
        --splits_dir  /path/to/v1.1/splits/flood_handlabeled \\
        --output_dir  /path/to/v1.1/splits \\
        --events Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA \\
        --val_fraction 0.1 \\
        --seed 42
"""

import argparse, random, re
from pathlib import Path

import pandas as pd


# Default event list — Sen1Floods11 v1.1 hand-labeled events.
DEFAULT_EVENTS = [
    "Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan",
    "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_dir",   required=True,
                   help="Source splits dir, e.g. .../splits/flood_handlabeled")
    p.add_argument("--output_dir",   required=True,
                   help="Where to write loro_<event>/ subdirs")
    p.add_argument("--events", nargs="+", default=DEFAULT_EVENTS,
                   help="Event names to use as held-out splits (default: all 11)")
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of non-event chips to use as validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_csv", default="flood_train_data.csv")
    p.add_argument("--valid_csv", default="flood_valid_data.csv")
    p.add_argument("--test_csv",  default="flood_test_data.csv")
    return p.parse_args()


def event_of(filename, events):
    """Pull the event name from a chip filename.

    The naming convention is ``<Event>_<chipid>_<modality>.tif`` so the
    event is everything before the first underscore. We match against the
    provided event list to avoid mis-parsing edge cases like 'Sri-Lanka'
    which contains a hyphen.
    """
    for ev in events:
        if filename.startswith(ev + "_"):
            return ev
    # Fallback: take the part before the first underscore. This will not
    # match the configured event list and the chip will be excluded from
    # any LORO split (logged as a warning).
    m = re.match(r"^([^_]+)_", filename)
    return m.group(1) if m else None


def load_all_chips(splits_dir, args):
    """Load every chip across train/val/test, keeping the (s1, label) pair."""
    chips = []
    for csv_name in (args.train_csv, args.valid_csv, args.test_csv):
        path = Path(splits_dir) / csv_name
        if not path.exists():
            print(f"[WARN] Source split CSV missing: {path}")
            continue
        df = pd.read_csv(path, header=None, names=["s1_file", "label_file"])
        chips.extend(df.itertuples(index=False, name=None))
    # De-duplicate (a chip should only ever appear once across the three CSVs)
    seen = set()
    unique = []
    for s1, lbl in chips:
        if s1 in seen: continue
        seen.add(s1); unique.append((s1, lbl))
    return unique


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["s1_file", "label_file"])
    df.to_csv(path, header=False, index=False)


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    chips = load_all_chips(args.splits_dir, args)
    print(f"Loaded {len(chips)} unique chips from {args.splits_dir}")

    # Bucket by event
    by_event = {ev: [] for ev in args.events}
    unmatched = 0
    for s1, lbl in chips:
        ev = event_of(s1, args.events)
        if ev in by_event:
            by_event[ev].append((s1, lbl))
        else:
            unmatched += 1
    if unmatched:
        print(f"[WARN] {unmatched} chips had no matching event name")

    print("\nChips per event:")
    for ev, items in by_event.items():
        print(f"  {ev:12s}  {len(items)}")

    out_root = Path(args.output_dir)
    summary = []

    for held_out in args.events:
        if not by_event[held_out]:
            print(f"[SKIP] {held_out} has 0 chips")
            continue

        # Test split = all chips for the held-out event
        test_rows = list(by_event[held_out])

        # Pool of train+val candidates = chips from all OTHER events
        pool = []
        for ev, items in by_event.items():
            if ev != held_out:
                pool.extend(items)

        # Deterministic shuffle, then carve val from the head
        pool_shuffled = pool[:]
        rng.shuffle(pool_shuffled)
        n_val = max(1, int(round(len(pool_shuffled) * args.val_fraction)))
        val_rows   = pool_shuffled[:n_val]
        train_rows = pool_shuffled[n_val:]

        # Write
        out_dir = out_root / f"loro_{held_out.lower()}"
        write_csv(train_rows, out_dir / "flood_train_data.csv")
        write_csv(val_rows,   out_dir / "flood_valid_data.csv")
        write_csv(test_rows,  out_dir / "flood_test_data.csv")

        summary.append({
            "event": held_out, "train": len(train_rows),
            "val": len(val_rows), "test": len(test_rows),
            "out": str(out_dir),
        })
        print(f"  loro_{held_out.lower():10s}  "
              f"train={len(train_rows):4d}  val={len(val_rows):3d}  "
              f"test={len(test_rows):3d}")

    print(f"\nWrote {len(summary)} LORO splits under {out_root}")


if __name__ == "__main__":
    main()
