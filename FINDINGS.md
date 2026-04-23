# Findings & Decisions Log

## EDA Round 1 — 2026-03-26

### Dataset Overview

| Subset | Chips | Labels |
|--------|-------|--------|
| Hand-labeled | 446 | Human-annotated binary masks |
| Weakly-labeled | 4,384 | Auto-generated (Otsu on S1 + water index on S2) |

**Splits (hand-labeled only):**

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 252 | Model training |
| Validation | 89 | Hyperparameter tuning |
| Test | 90 | Final evaluation |
| Bolivia | 15 | Held-out region (cross-region generalization) |

---

### Finding 1: Severe Class Imbalance

- **Mean water fraction per chip: 10.7%**
- Most chips have <5% water pixels — the histogram is heavily right-skewed
- Some chips are nearly 100% water (Mekong, Nigeria), others have almost none (USA)

**Decision:** Must use class-weighted loss (e.g., weighted BCE or Dice loss) during training. Plain cross-entropy will bias the model toward predicting everything as non-water.

---

### Finding 2: Geographic & Country Imbalance

- **13 countries** across 6 continents, but distribution is uneven:
  - USA, India, Paraguay — ~65-70 chips each (most represented)
  - Ghana — ~55 chips
  - Bolivia, Nigeria — ~15 chips each (least represented)
- Bolivia is held out entirely as a separate generalization test set (15 chips)

**Decision:** Consider stratified sampling by country during training to prevent the model from overfitting to USA/India landscapes. Cross-region generalization (milestone 10) will test on Bolivia.

---

### Finding 3: Water Fraction Varies Widely by Country

| Country | Mean Water Fraction | Notes |
|---------|-------------------|-------|
| Mekong | ~23% | River delta — lots of water |
| Nigeria | ~21% | Large flood extent |
| Bolivia | ~17% | Moderate flooding |
| Spain | ~15% | |
| India | ~13% | |
| Sri-Lanka | ~11% | |
| Paraguay | ~10% | |
| Pakistan | ~7% | |
| Ghana | ~5% | Sparse flooding in chips |
| Somalia | ~5% | |
| USA | ~3% | Most chips are mostly dry land |

**Decision:** Per-country water fraction is useful context for understanding model errors later. If the model fails on low-water-fraction regions, augmentation or oversampling those chips may help.

---

### Finding 4: S1 (SAR) Band Value Ranges

From 50-chip sample:

| Band | Range | Typical Range (2nd-98th pct) | Unit |
|------|-------|------------------------------|------|
| VV | ~ -55 to +10 | -20 to -3 | dB |
| VH | ~ -55 to -5 | -25 to -10 | dB |

- Both distributions are roughly Gaussian (in dB space)
- Mean shows `nan` in the plot — likely due to mixed float/list types in the aggregation (cosmetic bug, not a data issue)
- Water pixels concentrate at the low end (< -20 dB for VV, < -25 dB for VH)

**Decision:** Normalize S1 bands using dataset-wide mean/std (z-score normalization). Need to compute exact statistics across the full training set. The dB values are already log-scale, so no additional log transform needed.

---

### Finding 5: S2 (Optical) Band Value Ranges

From single-chip inspection (Ghana_103272):

| Band | Name | Min | Max | Mean |
|------|------|-----|-----|------|
| B1 | Coastal Aerosol | 1389 | 2955 | 1722 |
| B2 | Blue | 1085 | 3219 | 1476 |
| B3 | Green | 1029 | 3263 | 1473 |
| B4 | Red | 685 | 3360 | 1274 |
| B8 | NIR | 1381 | 4929 | 3104 |
| B11 | SWIR-1 | 1298 | 5045 | 2449 |
| B12 | SWIR-2 | 698 | 3414 | 1410 |

- Values are surface reflectance (int16), scaled by 10,000 in standard Sentinel-2 L2A
- SWIR bands (B11, B12) are particularly useful for water detection — water absorbs SWIR strongly

**Decision:** Normalize S2 per-band using training set mean/std. Consider using a subset of bands if 13-band input is too heavy (priority bands: B2, B3, B4, B8, B11, B12 — RGB + NIR + SWIR).

---

### Finding 6: Label Encoding

- **Values:** -1 (nodata/ignore), 0 (non-water), 1 (water)
- Nodata pixels exist in most chips (cloud cover, scan edges)

**Decision:** Mask out -1 pixels in the loss function using `ignore_index=-1`. These should not contribute to gradient updates or metric computation.

---

### Finding 7: Chip Metadata & CRS

- CRS: EPSG:4326 (WGS84 lat/lon)
- Resolution: ~10m/pixel (0.0000898° per pixel)
- Chip size: 512 × 512 = 262,144 pixels per chip (~5.1 km × 5.1 km)

**Decision:** When integrating DEM (Copernicus GLO-30 at 30m), will need to resample DEM to match 10m chip grid. Use bilinear interpolation for elevation, nearest-neighbor for slope/aspect.

---

### Finding 8: macOS Resource Fork Files

- Data copied from Mac left `._*.tif` binary files alongside every real `.tif`
- These break any `glob("*.tif")` pattern

**Decision:** All code that globs `.tif` files must filter with `not f.name.startswith(".")`. Applied throughout the notebook; must carry this into `src/data/` dataset classes.

---

### Finding 9: Geographic Distribution

- 13 flood events across 6 continents (South America, Africa, Asia, Europe, North America, Southeast Asia)
- Good geographic diversity, but each "country" is a single flood event — the model sees one type of terrain/climate per country

**Decision:** This is both a strength (diverse geography) and a risk (one event per region means limited within-region variation). Augmentation (flips, rotations, brightness jitter) is essential.

---

## Open Questions

1. **Should we use weakly-labeled data?** 4,384 extra chips available. Could pretrain on weak labels, then fine-tune on hand-labeled. Adds complexity — defer to after baseline models work.
2. **Which S2 bands matter most?** 13 bands is a lot. Ablation study (milestone 9) should test band subsets.
3. **DEM acquisition:** Need to download Copernicus DEM tiles for all 446 chip extents. Script needed.
4. **Normalization strategy:** Need to compute per-band mean/std across the full 252-chip training set for both S1 and S2.

---

## Decision Summary

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Use weighted loss (Dice or weighted BCE) | 10.7% mean water fraction — severe imbalance |
| D2 | Stratified sampling by country | Uneven country distribution (15-70 chips) |
| D3 | Z-score normalize S1 in dB space | Already log-scale, roughly Gaussian |
| D4 | Z-score normalize S2 per-band | Int16 reflectance values, varying ranges |
| D5 | Ignore label=-1 in loss | Nodata pixels must not affect training |
| D6 | Resample DEM to 10m to match chips | DEM is 30m, chips are 10m |
| D7 | Filter `._` files in all data loading | macOS resource forks break readers |
| D8 | Use augmentation (flips, rotations) | Limited within-region variation |
| D9 | Defer weakly-labeled data to later | Focus on hand-labeled baseline first |

---

## Baseline Implementation — 2026-03-26

### Original Paper Setup (from cloudtostreet/Sen1Floods11 repo)

We replicated the exact training setup from Bonafilia et al. (2020):

| Setting | Value | Source |
|---------|-------|--------|
| Model | FCN-ResNet50 (torchvision) | Train.ipynb |
| Input | S1 VV/VH only (2 channels) | Train.ipynb |
| First conv | Modified from 3→2 input channels | Train.ipynb |
| Normalization | BatchNorm → GroupNorm | Train.ipynb |
| S1 preprocessing | Clip [-50, 1] dB → scale to [0, 1] | Train.ipynb |
| Channel norm | mean=[0.6851, 0.5235], std=[0.0820, 0.1102] | Train.ipynb |
| Label nodata | Remap -1 → 255 (ignore_index) | Train.ipynb |
| Loss | CrossEntropyLoss(weight=[1, 8], ignore_index=255) | Train.ipynb |
| Optimizer | AdamW, lr=5e-4 | Train.ipynb |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10 epochs) | Train.ipynb |
| Epochs | 100 | Train.ipynb |
| Crop | Random 256×256 from 512×512 | Train.ipynb |
| Augmentation | Random H-flip + V-flip | Train.ipynb |
| Val inference | Full 512×512 (no crop) | Train.ipynb |

### Files Implemented

```
src/data/dataset.py          — Sen1Floods11 PyTorch Dataset + DataLoader factory
src/utils/metrics.py         — IoU, Dice, Precision, Recall, F1, Accuracy, MetricAccumulator
src/models/otsu_baseline.py  — Otsu thresholding on S1 VH (classical baseline)
src/models/fcn_baseline.py   — FCN-ResNet50 with 2-ch input + GroupNorm (paper baseline)
scripts/train_fcn_baseline.py — End-to-end training script with logging + checkpointing
scripts/evaluate.py          — Evaluate any model (otsu or fcn) on any split
notebooks/02_otsu_baseline.ipynb — Run + visualize Otsu results
```

### How to Run

```bash
# 1. Otsu baseline (no training needed)
python scripts/evaluate.py \
    --model otsu \
    --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
    --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
    --split test

# 2. Train FCN baseline
python scripts/train_fcn_baseline.py \
    --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
    --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
    --epochs 100 --batch_size 8

# 3. Evaluate trained FCN
python scripts/evaluate.py \
    --model fcn \
    --checkpoint results/checkpoints/fcn_baseline_best.pt \
    --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
    --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
    --split test
```

### Decision D10: Follow Paper Preprocessing Exactly

Rather than inventing our own normalization (D3 said z-score), we follow the paper's
exact clip-and-scale approach: `clip([-50, 1]) → (x + 50) / 51 → channel normalize`.
This ensures our baseline numbers are comparable to published results. We can experiment
with different normalization later during ablations.

### Baseline Results

| Model | Split | IoU | Dice | Precision | Recall | Accuracy |
|-------|-------|-----|------|-----------|--------|----------|
| Otsu (VH) | Train | 0.1867 | 0.3146 | 0.1996 | 0.7426 | 0.6925 |
| Otsu (VH) | Val | 0.2199 | 0.3605 | 0.2365 | 0.7582 | 0.7034 |
| Otsu (VH) | **Test** | **0.2299** | **0.3738** | **0.2532** | **0.7143** | **0.7007** |
| Otsu (VH) | Bolivia | 0.4032 | 0.5747 | 0.4322 | 0.8572 | 0.7987 |
| FCN-ResNet50 | Val (best) | 0.6145 | — | — | — | — |
| FCN-ResNet50 | **Test** | **0.6380** | **0.7790** | **0.7926** | **0.7659** | **0.9456** |
| FCN-ResNet50 | Bolivia | — | — | — | — | — |

### Otsu Baseline Analysis (2026-03-26)

**Finding 10: Otsu has high recall but terrible precision**
- Recall ~71-86%: it catches most water pixels (low VH = water works)
- Precision ~20-43%: massive false positive rate — it labels shadows, dark soil, and urban areas as water
- This is exactly the failure mode the proposal predicted: "mountain shadows mimic water in SAR"

**Finding 11: Bolivia outperforms other regions significantly**
- Bolivia IoU (0.40) is nearly 2x the test IoU (0.23)
- Bolivia has large, contiguous flood areas with clean water surfaces → Otsu's sweet spot
- Regions with fragmented flooding or terrain complexity (Ghana, Somalia) drag the average down

**Finding 12: Bimodal per-chip IoU distribution**
- ~35 chips have IoU near 0 (no water or total Otsu failure)
- ~15 chips have IoU > 0.6 (Otsu works well on clean flood scenes)
- The middle is sparse — Otsu either works or it doesn't, no graceful degradation

**Finding 13: Test IoU = 0.23 is the floor to beat**
- The FCN-ResNet50 should significantly exceed this
- Published FCNN results from the paper report ~0.30-0.50 IoU depending on configuration
- Our multi-modal fusion model should target >0.55 IoU

---

## Phase 2: Cross-Attention Fusion U-Net (S1 + S2) — 2026-03-26

### Architecture

```
S1 (2ch) → ResNet34 Encoder → features at 4 scales ─┐
                                                      ├→ Cross-Attention → Concat+Fuse → U-Net Decoder → (B, 2, H, W)
S2 (13ch) → ResNet34 Encoder → features at 4 scales ─┘
```

**Encoder scales:**

| Scale | Resolution | Channels | Cross-Attention Heads |
|-------|-----------|----------|----------------------|
| 0 | H/4 × W/4 | 64 | 4 |
| 1 | H/8 × W/8 | 128 | 4 |
| 2 | H/16 × W/16 | 256 | 4 |
| 3 | H/32 × W/32 | 512 | 4 |

**Cross-attention mechanism:**
- Bi-directional: S1 attends to S2 AND S2 attends to S1 at each scale
- Reduced projection dimension (25% of channel dim) for memory efficiency
- Residual connections: `A_out = A + Attn(Q=A, K=B, V=B)`
- After attention, features are concatenated and projected back: `Fused = Conv1x1(Cat(S1_att, S2_att))`

**Decoder:** Standard U-Net decoder with skip connections from fused features at each scale.

### S2 Preprocessing

- Divide by 10,000 (Sentinel-2 L2A reflectance scaling factor)
- Clip to [0, 10000] before scaling
- Z-score normalize per-band using training set statistics

**S2 normalization stats (computed on 252 training chips):**

| Band | Name | Mean | Std |
|------|------|------|-----|
| B1 | Coastal Aerosol | 0.1627 | 0.0700 |
| B2 | Blue | 0.1396 | 0.0739 |
| B3 | Green | 0.1364 | 0.0735 |
| B4 | Red | 0.1218 | 0.0865 |
| B5 | Red Edge 1 | 0.1466 | 0.0777 |
| B6 | Red Edge 2 | 0.2387 | 0.0921 |
| B7 | Red Edge 3 | 0.2846 | 0.1084 |
| B8 | NIR | 0.2623 | 0.1023 |
| B8A | NIR Narrow | 0.3077 | 0.1196 |
| B9 | Water Vapor | 0.0487 | 0.0337 |
| B10 | Cirrus | 0.0064 | 0.0144 |
| B11 | SWIR-1 | 0.2031 | 0.0981 |
| B12 | SWIR-2 | 0.1179 | 0.0765 |

### Training Setup

| Setting | Baseline (FCN) | Fusion U-Net | Rationale |
|---------|----------------|--------------|-----------|
| Backbone | ResNet50 | 2× ResNet34 | Lighter per-encoder, fits in 8GB VRAM |
| Input | S1 only (2ch) | S1 (2ch) + S2 (13ch) | Multi-modal fusion |
| Batch size | 8 | 4 | Dual encoders use more memory |
| Mixed precision | No | Yes (AMP) | Required for 8GB VRAM |
| Optimizer | AdamW, lr=5e-4 | AdamW, lr=5e-4 | Same for fair comparison |
| Scheduler | CosineAnnealingWarmRestarts | Same | Same for fair comparison |
| Loss | CE(weight=[1,8], ignore=255) | Same | Same for fair comparison |
| Crop | 256×256 | 256×256 | Same |
| Epochs | 100 | 100 | Same |
| Parameters | ~33M | ~48.5M | +47% from dual encoder + attention |

### Files Implemented

```
src/data/dataset.py              — Added Sen1Floods11MultiModal + get_multimodal_dataloaders()
src/models/fusion_unet.py        — FusionUNet (dual ResNet34 + CrossAttention + U-Net decoder)
scripts/train_fusion.py          — Training script with AMP support
```

### How to Run

```bash
python scripts/train_fusion.py \
    --data_root F:/Sen1Flood1/v1.1/data/flood_events/HandLabeled \
    --splits_dir F:/Sen1Flood1/v1.1/splits/flood_handlabeled \
    --epochs 100 --batch_size 4
```

### Decision D11: Cross-Attention over Early Fusion

Why not just concatenate S1+S2 into 15 channels (early fusion)?
- Early fusion forces the first conv layer to learn joint S1+S2 features from scratch
- S1 (SAR, dB values) and S2 (optical, reflectance) have fundamentally different distributions
- Cross-attention lets each modality selectively attend to relevant regions of the other
- SAR sees through clouds but has speckle noise; optical has clearer spatial patterns but fails under clouds
- Attention learns **when** to trust each modality per-pixel

### Decision D12: ResNet34 over ResNet50 for Dual Encoders

- Two ResNet50 encoders = ~50M encoder params; two ResNet34 = ~42M
- ResNet34 uses BasicBlock (2 convs) vs Bottleneck (3 convs) → less memory per layer
- With AMP + batch_size=4 + 256×256 crops, fits in 8GB VRAM
- ResNet34 has proven sufficient for satellite segmentation (SegFormer, etc.)

### Decision D13: Use All 13 S2 Bands

- Rather than selecting a subset of bands, feed all 13 to the encoder
- The model can learn to ignore uninformative bands (B9 water vapor, B10 cirrus)
- SWIR bands (B11, B12) are critical for water detection — water strongly absorbs SWIR
- NIR (B8) helps distinguish vegetation from water
- Ablation in Phase 3 will test band subsets
