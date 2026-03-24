# Multi-Modal Flood Segmentation

**ECE 574 — Computer Vision Project**

Cross-attention fusion of Sentinel-1 (SAR), Sentinel-2 (multispectral), and DEM data
for binary flood segmentation with uncertainty estimation.

## Team
- Hamna — Remote sensing / geospatial pipeline
- Eric — Deep learning / model architecture

## Dataset
- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., CVPR 2020)
- [Copernicus DEM GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)

## Project Structure

```
flood-segmentation/
├── configs/            # Training and experiment configs (YAML)
├── data/
│   ├── raw/            # Original downloads (S1, S2, labels, DEM)
│   ├── processed/      # Aligned, normalized, ready-to-train chips
│   └── splits/         # Train/val/test split definitions
├── notebooks/          # Exploration and visualization
├── results/
│   ├── checkpoints/    # Saved model weights
│   ├── figures/        # Plots, maps, uncertainty visualizations
│   └── logs/           # Training logs, metrics
├── scripts/            # Download, preprocessing, eval scripts
├── src/
│   ├── data/           # Dataset classes, transforms, dataloaders
│   ├── models/         # U-Net, attention modules, baselines
│   └── utils/          # Metrics, visualization, helpers
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone <this-repo>
cd flood-segmentation
pip install -r requirements.txt
```

## Milestones

| # | Task | Deadline |
|---|------|----------|
| 1 | Literature review & baseline replication | 02/09/26 |
| 2 | Download & preprocess Sen1Floods11 + DEM | 02/16/26 |
| 3 | Classical baseline: Otsu on S1 VH | 02/23/26 |
| 4 | Baseline U-Net on S1-only | 03/02/26 |
| 5 | Multi-modal fusion module (S1+S2+DEM) | 03/09/26 |
| 6 | Cross-modal attention in U-Net encoder | 03/16/26 |
| 7 | Train & evaluate attention-fusion model | 03/23/26 |
| 8 | MC Dropout uncertainty estimation | 03/30/26 |
| 9 | Ablation studies | 04/06/26 |
| 10 | Cross-region generalization testing | 04/13/26 |
| 11 | Final report figures & write-up | 04/20/26 |
| 12 | Project presentation | 04/27/26 |
| 13 | Final report delivery | 05/05/26 |
