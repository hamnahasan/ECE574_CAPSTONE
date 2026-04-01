# Multi-Modal Flood Segmentation

**ECE 574 — Computer Vision Project**

Binary flood segmentation from Sentinel-1 SAR and Sentinel-2 multispectral imagery using a dual-encoder U-Net with cross-attention fusion and MC Dropout uncertainty estimation.

## Team
- **Hamna** — Remote sensing, geospatial pipeline
- **Eric** — Deep learning, model architecture

## Dataset
- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., CVPR 2020)
  - 446 hand-labeled chips, 512×512, 11 flood events across 6 continents
  - Sentinel-1 (VV/VH SAR) + Sentinel-2 (13-band multispectral) + binary flood labels
- [Copernicus DEM GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) (planned)

## Results

| Model | Input | Test IoU | Test Dice | Notes |
|-------|-------|----------|-----------|-------|
| Otsu Thresholding | S1 VH | 0.230 | — | Classical baseline |
| FCN-ResNet50 | S1 (2ch) | **0.638** | 0.779 | Deep learning baseline |
| Fusion U-Net (cross-attn) | S1 + S2 | in progress | — | Val IoU 0.706 @ epoch 1 |

## Architecture

```
S1 (2ch)  → ResNet34 Encoder ─┐
                               ├─ Cross-Attention at 4 scales ─→ U-Net Decoder → Segmentation
S2 (13ch) → ResNet34 Encoder ─┘
```

**Cross-attention:** At each of the 4 encoder scales, S1 features attend to S2 and vice versa. This lets the model learn when to trust SAR (clouds, night) vs optical (clearer spatial boundaries) per spatial region.

**Uncertainty (Phase 3):** MC Dropout at inference produces per-pixel uncertainty maps for disaster response decision support.

## Project Structure

```
configs/            — Training and experiment configs (YAML)
data/
  raw/              — Sen1Floods11 chips (gitignored, local at F:/Sen1Flood1)
  splits/           — Train/val/test split definitions
notebooks/          — EDA and visualization notebooks
results/
  checkpoints/      — Saved model weights (.pt)
  figures/          — Training curves, error maps
  logs/             — Training histories (JSON)
scripts/            — Training, evaluation, download scripts
src/
  data/dataset.py   — Sen1Floods11 + Sen1Floods11MultiModal dataset classes
  models/
    otsu_baseline.py   — Classical Otsu thresholding
    fcn_baseline.py    — FCN-ResNet50 (S1-only baseline)
    fusion_unet.py     — Dual-encoder U-Net with cross-attention
  utils/metrics.py  — IoU, Dice, confusion matrix, MetricAccumulator
requirements.txt
```

## Setup

```bash
git clone https://github.com/hamnahasan/ECE574_CAPSTONE
cd ECE574_CAPSTONE
python -m venv venv
pip install -r requirements.txt
```

Data is expected at `F:/Sen1Flood1/v1.1/` or set paths manually in each script.

## Training

**FCN Baseline (S1 only):**
```bash
python scripts/train_fcn_baseline.py \
    --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
    --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled
```

**Fusion U-Net (S1 + S2):**
```bash
python scripts/train_fusion.py \
    --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
    --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
    --epochs 100 --batch_size 4
```

## Milestones

| # | Task | Deadline | Status |
|---|------|----------|--------|
| 1 | Literature review & baseline replication | 02/09/26 | Done |
| 2 | Download & preprocess Sen1Floods11 | 02/16/26 | Done |
| 3 | Classical baseline: Otsu on S1 VH | 02/23/26 | Done — IoU 0.230 |
| 4 | Baseline U-Net (S1-only, FCN-ResNet50) | 03/02/26 | Done — IoU 0.638 |
| 5 | Multi-modal fusion S1+S2 | 03/09/26 | Done — architecture implemented |
| 6 | Cross-modal attention U-Net | 03/16/26 | In progress — training |
| 7 | Train & evaluate attention-fusion model | 03/23/26 | In progress |
| 8 | MC Dropout uncertainty estimation | 03/30/26 | Pending |
| 9 | Ablation studies | 04/06/26 | Pending |
| 10 | Cross-region generalization testing | 04/13/26 | Pending |
| 11 | Final report figures & write-up | 04/20/26 | Pending |
| 12 | Project presentation | 04/27/26 | Pending |
| 13 | Final report delivery | 05/05/26 | Pending |

## References

1. Bonafilia et al. (2020) — Sen1Floods11, CVPR Workshops
2. Bai et al. (2021) — S1+S2 fusion with BASNet
3. Ronneberger et al. (2015) — U-Net
4. Gal & Ghahramani (2016) — MC Dropout uncertainty
