# SLURM Job Scripts for Isaac (UTK ISAAC-NG)

These scripts launch training on Isaac with auto-resume so long runs survive
the 24hr wall-time limit.

## Setup (first time only)

```bash
# 1. Clone the repo on Isaac
cd /lustre/isaac24/scratch/$USER
git clone https://github.com/hamnahasan/ECE574_CAPSTONE.git
cd ECE574_CAPSTONE

# 2. Build Python environment (see ../scripts/setup_isaac.sh)
bash scripts/setup_isaac.sh

# 3. Confirm data is transferred to /lustre/isaac24/scratch/$USER/sen1flood1/
ls /lustre/isaac24/scratch/$USER/sen1flood1/v1.1/data/flood_events/HandLabeled/
```

## Launch All Experiments

```bash
# Tier 1 — Core results (2 jobs, ~40hr each split across 2 SLURM requeues)
sbatch slurm/train_fusion.sbatch       # Phase 2: S1+S2 cross-attention
sbatch slurm/train_trimodal.sbatch     # Phase 3: S1+S2+DEM tri-modal

# Tier 2 — Ablation study (7 variants in parallel via SLURM array)
sbatch slurm/train_ablation_array.sbatch

# Monitor
squeue -u $USER
tail -f logs/slurm-*.out
```

## How Auto-Resume Works

Every SLURM script passes `--auto_resume` to the training script. When a job
hits the 24hr wall-time:

1. SLURM sends SIGTERM, then SIGKILL
2. Training script dies mid-epoch (atomic checkpoint write prevents corruption)
3. SLURM auto-requeues the same job via `#SBATCH --requeue`
4. On restart, `--auto_resume` picks up `<run_name>_latest.pt`
5. Resumes from the exact next epoch with same LR schedule + RNG state

No manual intervention needed. A 100-epoch run just takes 2 SLURM wall-times.

## Files

| File | Purpose |
|------|---------|
| `train_fusion.sbatch`          | Phase 2 — 100 epochs, auto-resume |
| `train_trimodal.sbatch`        | Phase 3 — 100 epochs, with modality dropout |
| `train_ablation_array.sbatch`  | 7 ablation runs in parallel |
| `eval_bolivia.sbatch`          | Cross-region generalization test |
| `mc_uncertainty.sbatch`        | MC Dropout + ECE on test set |
