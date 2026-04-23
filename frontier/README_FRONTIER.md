# Frontier Slurm Playbook

Additional Frontier/OLCF files for running the flood segmentation experiments.
These files are intentionally separate from the ISAAC files.

## 1. Fill in placeholders

Edit these values before submitting jobs:

- In every `*.sbatch`: replace `PROJECT_ID_HERE` in `#SBATCH -A PROJECT_ID_HERE`.
- In `frontier/frontier_env.sh`: set `PROJECT_DIR`, `DATA_ROOT`, and `SPLITS_DIR`.

Expected data layout:

```bash
$DATA_ROOT/
  S1Hand/
  S2Hand/
  DEMHand/
  LabelHand/

$SPLITS_DIR/
  flood_train_data.csv
  flood_valid_data.csv
  flood_test_data.csv
  flood_bolivia_data.csv
```

## 2. One-time environment setup

Run this from the repo root on Frontier:

```bash
bash frontier/setup_frontier.sh
```

The setup creates a conda environment under Frontier scratch, installs PyTorch
for ROCm, and installs the project dependencies.

## 3. Pre-flight check

Submit the sanity-check job first:

```bash
sbatch frontier/verify_setup_frontier.sbatch
```

Only launch training after the verify job prints all `[PASS]` checks.

## 4. Training

Launch core models:

```bash
sbatch frontier/train_fcn_frontier.sbatch
sbatch frontier/train_fusion_frontier.sbatch
sbatch frontier/train_trimodal_frontier.sbatch
```

Launch ablations:

```bash
sbatch frontier/train_ablation_array_frontier.sbatch
```

Optional packed Frontier jobs, usually better when node allocations are
exclusive:

```bash
sbatch frontier/train_core_packed_frontier.sbatch
sbatch frontier/train_ablation_packed_frontier.sbatch
```

The packed jobs run multiple single-GPU Python processes inside one node
allocation, with Slurm assigning one GPU per process.

Each training job uses `--auto_resume` and writes checkpoints/logs under
`results/`.

## 5. Post-training jobs

```bash
sbatch frontier/eval_bolivia_frontier.sbatch
sbatch frontier/mc_uncertainty_frontier.sbatch
sbatch frontier/compile_results_frontier.sbatch
```

## 6. Aggregate results

You can run this as a Slurm job:

```bash
sbatch frontier/compile_results_frontier.sbatch
```

Or run it inside an interactive/login session after activating the Frontier env:

```bash
conda activate <PROJECT_DIR>/.frontier_envs/floodseg
python scripts/compile_results.py

# Produces:
#   results/logs/all_results.csv    - every metric, every experiment
#   results/logs/all_results.md     - paper-ready Markdown tables
```

## 7. Pull results back locally for notebooks

From your local machine, pull the Frontier `results/` folder back into this repo.
Use `rsync` instead of `scp` because it can resume large checkpoint transfers.

```bash
# Pull logs, figures, and checkpoints, but not raw data.
rsync -avz --progress \
    <frontier_user>@frontier.olcf.ornl.gov:<PROJECT_DIR>/results/ \
    results/
```

Example after placeholders are known:

```bash
rsync -avz --progress \
    her_username@frontier.olcf.ornl.gov:/lustre/orion/PROJECT_ID/scratch/her_username/ECE574_CAPSTONE/results/ \
    results/
```

To pull only lightweight logs and figures first:

```bash
rsync -avz --progress \
    --include='logs/***' \
    --include='figures/***' \
    --exclude='checkpoints/***' \
    <frontier_user>@frontier.olcf.ornl.gov:<PROJECT_DIR>/results/ \
    results/
```

To pull checkpoints later:

```bash
rsync -avz --progress \
    <frontier_user>@frontier.olcf.ornl.gov:<PROJECT_DIR>/results/checkpoints/ \
    results/checkpoints/
```

## Notes

Frontier uses AMD GPUs through ROCm. PyTorch still reports devices through the
`torch.cuda` API on ROCm builds, so the existing training scripts can run
unchanged.
