# Isaac Playbook — Run all experiments in 3 days

Quick reference for running the full experimental suite on UTK ISAAC-NG.

## 0. Prerequisites (you already have these)

- [x] Data transferred to `/lustre/isaac24/scratch/eopoku2/sen1flood1/`
- [x] Code in GitHub at `github.com/hamnahasan/ECE574_CAPSTONE`

## 1. Environment Setup (once)

```bash
ssh eopoku2@login.isaac.utk.edu
cd /lustre/isaac24/scratch/$USER

git clone https://github.com/hamnahasan/ECE574_CAPSTONE.git
cd ECE574_CAPSTONE
git pull

bash scripts/setup_isaac.sh         # ~10 min: creates 'floodseg' conda env
```

## 2. Pre-flight sanity check (REQUIRED before any training)

This catches silent bugs that would invalidate paper results:
data alignment, augmentation consistency, model forward/backward pass,
gradient flow, save/load round-trip.

```bash
source activate floodseg
python scripts/verify_setup.py \
    --data_root  /lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled \
    --splits_dir /lustre/isaac24/scratch/$USER/sen1flood1/v1.1/splits/flood_handlabeled
```

Expect 5 sections of `[PASS]` checks. Exit code 0 means safe to launch.
If anything fails, **do not submit SLURM jobs** — fix first.

## 3. Tier 1 — Core models (launch together, ~3 GPUs in parallel)

```bash
sbatch slurm/train_fcn.sbatch        # Phase 1: FCN-ResNet50 S1 baseline (~20hr)
sbatch slurm/train_fusion.sbatch     # Phase 2: S1+S2 cross-attention (~40hr, 2 wall-times)
sbatch slurm/train_trimodal.sbatch   # Phase 3: S1+S2+DEM tri-modal (~40hr, 2 wall-times)
```

All three auto-requeue when the 24hr wall-time hits and resume from the
latest checkpoint.

## 4. Tier 2 — Ablation array (launch together, 7 GPUs in parallel)

```bash
sbatch slurm/train_ablation_array.sbatch
```

Submits 7 jobs at once: `s1`, `s2`, `dem`, `s1_s2`, `s1_dem`, `s2_dem`, `s1_s2_dem`.
Each runs independently; Isaac schedules them on different GPUs as available.

## 5. Monitor

```bash
squeue -u $USER                      # queue state
watch -n 30 'squeue -u $USER'

# Follow a specific log
tail -f logs/slurm-fusion_s1s2-*.out
tail -f logs/slurm-trimodal_s1s2dem-*.out

# See current progress on every job
for f in logs/slurm-*.out; do echo "--- $f ---"; tail -3 $f; done
```

### How to read the training output

Every epoch line shows:

```
Epoch  42/100 | Train: 0.1234 (IoU 0.812) | Val: 0.1567 (IoU 0.749, Dice 0.857) | Gap: +0.063 | 1234.5s
```

| Field | Meaning |
|-------|---------|
| `Train: 0.1234` | Training loss (lower is better) |
| `(IoU 0.812)` | **Train IoU** — accuracy on the augmented training set |
| `Val: 0.1567` | Validation loss |
| `(IoU 0.749, Dice 0.857)` | Val IoU and Dice — what we optimize against |
| `Gap: +0.063` | **train_iou - val_iou — the overfitting signal** |

**Watch the Gap.** Healthy training: gap stays small (< 0.10) and val_iou
keeps rising. Overfitting: gap grows above 0.15 while val_iou plateaus or
drops. We DON'T early-stop automatically — final reporting always uses
the best-by-val-IoU checkpoint regardless of when overfitting starts.

## 6. Tier 3 — Run after Tier 1+2 finishes

```bash
sbatch slurm/eval_bolivia.sbatch     # Cross-region generalization
sbatch slurm/mc_uncertainty.sbatch   # MC Dropout + ECE
```

## 7. Aggregate results

```bash
source activate floodseg
python scripts/compile_results.py

# Produces:
#   results/logs/all_results.csv    — every metric, every experiment
#   results/logs/all_results.md     — paper-ready Markdown tables
```

## 8. Pull results back locally for notebooks

From your local machine:

```bash
# Pull only the logs + checkpoints (not the raw data)
rsync -avz --progress \
    eopoku2@login.isaac.utk.edu:/lustre/isaac24/scratch/eopoku2/ECE574_CAPSTONE/results/ \
    results/

# Then run notebooks locally
jupyter notebook notebooks/05_ablation_and_uncertainty.ipynb
```

The notebook now plots **train IoU vs val IoU per epoch** so you can show
explicitly that the model is not overfitting. This is the figure reviewers
expect to see for every learning curve.

---

## Expected Timeline

| Day | Jobs running | Status |
|-----|--------------|--------|
| 0 (tonight) | — | Code + SLURM scripts pushed |
| 1 morning   | FCN + Fusion + TriModal + 7 ablations = **10 GPUs** | First 24hr window |
| 2 morning   | Long jobs auto-requeued | Second 24hr window; most finish |
| 2 evening   | Bolivia + MC uncertainty | Short jobs (~1-2hr) |
| 3           | `compile_results.py`, rsync, notebooks | Paper-ready tables and figures |

## Defending the experimental design

Reviewers will ask about three things — pre-empt every one:

| Concern | Where it's addressed |
|---------|---------------------|
| Augmentations? | [AUGMENTATIONS.md](AUGMENTATIONS.md) — every aug listed with justification |
| Overfitting? | `train_iou` vs `val_iou` per epoch in `*_history.json`; plotted in notebook 05 |
| Fair comparison? | All models share same loss, scheduler, augmentation, splits, seed |
| Reproducibility? | `--seed 42`, RNG state checkpointed, AUGMENTATIONS.md lists all stats |
| DEM aligned correctly? | Section 10 of `01_explore_data.ipynb` with side-by-side viz |
| Cross-region? | Bolivia held out, evaluated by `eval_bolivia.sbatch` |
| Calibration? | MC Dropout + ECE + reliability diagram from `mc_uncertainty.sbatch` |

## Troubleshooting

**`verify_setup.py` fails on augmentation check:**
This is the most dangerous failure — means crops/flips are inconsistent
across modalities. Stop and inspect `Sen1Floods11TriModal.__getitem__`.
Do NOT launch SLURM jobs.

**Job fails immediately with "module not found":**
Conda env not activating. Use `source activate floodseg` (Isaac uses
old-style activation, not `conda activate`).

**Auto-resume starts from scratch:**
Check `results/checkpoints/<run_name>_latest.pt` exists. If not, the
first epoch never completed — inspect the SLURM error log.

**OOM during training:**
Lower `--batch_size` in the sbatch file (8 → 4, or 4 → 2). Safe because
we use gradient clipping and GroupNorm (batch-size-independent).

**Modify a running experiment:**
`scancel <jobid>`, edit the script, `sbatch` again. `--auto_resume`
picks up exactly where it stopped.

**Train IoU >> Val IoU growing every epoch (overfitting signal):**
Check the Gap column. If it climbs past 0.20 with val_iou stagnating,
the model is overfitting. Remediation options:
- Lower learning rate (`--lr 1e-4`)
- Increase modality dropout (`--mod_dropout 0.2`)
- Stop training; the best-by-val checkpoint is already saved
