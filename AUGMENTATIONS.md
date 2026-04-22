# Data Augmentation Reference

Single source of truth for what augmentations are applied during training.
**This document is the answer to any reviewer asking "what augmentations did you use?"**

## Summary

| Augmentation | Train | Val | Test | Bolivia |
|--------------|:-----:|:---:|:----:|:-------:|
| Random crop 256×256        | yes | no | no | no |
| Random horizontal flip 50% | yes | no | no | no |
| Random vertical flip 50%   | yes | no | no | no |
| Z-score normalization      | yes | yes | yes | yes |
| Modality dropout (TriModal only) | yes | no | no | no |

Validation, test, and Bolivia evaluation use **full 512×512 chips with no
augmentation**. Only the training set sees random crops and flips.

## Implementation Details

### Random crop (training only)

Crop to 256×256 from the original 512×512 chip. Crop position is sampled
uniformly: `i ∈ [0, 256], j ∈ [0, 256]`.

**Why a smaller crop than the original?** Doubles the effective batch size
on 8GB VRAM. Original Sen1Floods11 paper uses 256×256 crops as well.

**Critical implementation detail:** The same `(i, j)` crop coordinates are
applied to S1, S2, DEM, and the label. Independent crops would silently
destroy training (predictions for one location supervised by labels from
another). See `verify_setup.py` check #2 for the test that catches this.

```python
# src/data/dataset.py
i, j, th, tw = Sen1Floods11._get_crop_params(h, w, self.crop_size)
s1    = s1[:,  i:i+th, j:j+tw]
s2    = s2[:,  i:i+th, j:j+tw]
dem   = dem[:, i:i+th, j:j+tw]
label = label[i:i+th, j:j+tw]
```

### Random horizontal flip (training only)

Applied with probability 0.5 to all modalities together using
`torchvision.transforms.functional.hflip`.

**Justification for floods:** Flood patterns have no preferred left-right
orientation (rivers can flow either direction in image coordinates), so
hflip is a label-preserving augmentation.

### Random vertical flip (training only)

Applied with probability 0.5. Independent of hflip.

**Justification:** Same as hflip — no preferred up-down orientation.
Together hflip + vflip yields 4 effective rotations (0°, 90° via vflip+hflip,
180°, 270°) without needing rotation augmentation.

**Why not arbitrary rotations?** Sentinel-1/2 chips have north-up
orientation. Arbitrary rotations would require interpolation that
introduces resampling artefacts in SAR speckle. We deliberately exclude
this to avoid changing the noise statistics of the training data.

### Z-score normalization (always applied)

Per-channel: `x = (x - mean) / std` where mean/std are precomputed from
the **training set only** (252 chips). Applied identically at train, val,
test, and Bolivia.

**Computed statistics:**

```python
# src/data/dataset.py
S1_MEAN  = [0.6851, 0.5235]                                   # VV, VH (after clip+scale)
S1_STD   = [0.0820, 0.1102]
S2_MEAN  = [0.1627, 0.1396, 0.1364, 0.1218, 0.1466, 0.2387,   # 13 bands (after /10000)
            0.2846, 0.2623, 0.3077, 0.0487, 0.0064, 0.2031, 0.1179]
S2_STD   = [0.0700, 0.0739, 0.0735, 0.0865, 0.0777, 0.0921,
            0.1084, 0.1023, 0.1196, 0.0337, 0.0144, 0.0981, 0.0765]
DEM_MEAN = [154.2425, 3.1475]                                 # elevation_m, slope_deg
DEM_STD  = [140.8741, 5.2020]
```

### Modality dropout (TriModal training only)

Each modality is independently zeroed with probability `--mod_dropout`
(default 0.1). Implemented in `scripts/train_trimodal.py::apply_modality_dropout`.

**Justification:** Floods are commonly observed under cloud cover (S2
saturates), and DEM tiles can be missing in some regions. Training with
random modality dropout forces the model to learn complementary
representations and remain robust at inference when a modality is degraded.
Also acts as additional regularization.

**Independent of standard augmentation pipeline.** Crop and flip happen
first; modality dropout is applied to the GPU tensors right before the
forward pass.

## What is deliberately NOT used

These augmentations were considered and rejected with reasoning:

| Augmentation | Why excluded |
|--------------|--------------|
| Random rotation | Resampling alters SAR speckle statistics |
| Color jitter / brightness | S1 is in dB; "brightness" has no physical meaning |
| Cutout / masked modeling | Would conflict with cloud/missing-data semantics |
| MixUp / CutMix | Mixes flood and non-flood pixels — destroys spatial labels |
| Elastic deformation | Common in medical imaging, but warps the geographic grid |
| Gaussian noise | SAR already has speckle; adding noise compounds it badly |
| Per-channel band dropout (within S2) | Could be interesting future work |

## Verifying augmentation correctness

Run the sanity check before any training:

```bash
python scripts/verify_setup.py \
    --data_root  /path/to/HandLabeled \
    --splits_dir /path/to/flood_handlabeled
```

Check #2 ("Augmentation consistency") confirms that flip and crop are
applied identically to all modalities — this is the silent bug that would
invalidate every result.

## Reproducing the exact training augmentations

All randomness is seeded by `--seed 42` (default in every training script).
With auto-resume enabled, the RNG state is checkpointed and restored, so
training resumes from the exact next augmentation it would have produced.

For paper Figure reproducibility, use `--seed 42` and verify the per-epoch
train_iou and val_iou match what's in the published `*_history.json` files.
