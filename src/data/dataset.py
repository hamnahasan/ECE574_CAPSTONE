"""Sen1Floods11 PyTorch dataset.

Supports two modes:
  1. S1-only (original baseline): loads VV/VH, clip/scale/normalize
  2. Multi-modal (S1 + S2): loads both modalities, returns them separately
     for dual-encoder architectures

Preprocessing follows the original paper for S1.
S2 is divided by 10000 (surface reflectance scaling) then z-score normalized.
"""

import numpy as np
import pandas as pd
import rasterio
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import random


# Precomputed on training set by original authors (after clip+scale)
S1_MEAN = [0.6851, 0.5235]
S1_STD = [0.0820, 0.1102]

# Precomputed on 252-chip training set (S2 / 10000)
S2_MEAN = [0.1627, 0.1396, 0.1364, 0.1218, 0.1466, 0.2387, 0.2846,
           0.2623, 0.3077, 0.0487, 0.0064, 0.2031, 0.1179]
S2_STD = [0.0700, 0.0739, 0.0735, 0.0865, 0.0777, 0.0921, 0.1084,
          0.1023, 0.1196, 0.0337, 0.0144, 0.0981, 0.0765]

# Precomputed on 252-chip training set (band1=elevation_m, band2=slope_deg)
DEM_MEAN = [154.2425, 3.1475]
DEM_STD  = [140.8741, 5.2020]


class Sen1Floods11(Dataset):
    """Sen1Floods11 dataset for S1-only flood segmentation.

    Args:
        split_csv: Path to CSV file with (s1_filename, label_filename) rows.
        s1_dir: Directory containing S1 .tif files.
        label_dir: Directory containing label .tif files.
        crop_size: If set, randomly crop to this size (training). None = full 512.
        augment: Whether to apply random flips (training only).
        normalize: Whether to apply channel-wise normalization.
    """

    def __init__(
        self,
        split_csv,
        s1_dir,
        label_dir,
        crop_size=256,
        augment=False,
        normalize=True,
    ):
        self.s1_dir = Path(s1_dir)
        self.label_dir = Path(label_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = normalize

        # Load split CSV (no header: s1_file, label_file)
        df = pd.read_csv(split_csv, header=None, names=["s1_file", "label_file"])
        self.samples = list(zip(df["s1_file"], df["label_file"]))

    def __len__(self):
        return len(self.samples)

    def _read_s1(self, path):
        """Read S1 chip, clip to [-50, 1], scale to [0, 1]."""
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (2, H, W)
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, -50, 1)
        data = (data + 50) / 51.0  # scale to [0, 1]
        return data

    def _read_label(self, path):
        """Read label, remap -1 (nodata) to 255 for ignore_index."""
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.int64)  # (H, W)
        data[data == -1] = 255
        return data

    def __getitem__(self, idx):
        s1_file, label_file = self.samples[idx]
        s1 = self._read_s1(self.s1_dir / s1_file)
        label = self._read_label(self.label_dir / label_file)

        # Convert to tensors
        s1 = torch.from_numpy(s1)        # (2, H, W)
        label = torch.from_numpy(label)   # (H, W)

        # Random crop
        if self.crop_size is not None:
            _, h, w = s1.shape
            i, j, th, tw = self._get_crop_params(h, w, self.crop_size)
            s1 = s1[:, i:i + th, j:j + tw]
            label = label[i:i + th, j:j + tw]

        # Random augmentation (flips)
        if self.augment:
            if random.random() > 0.5:
                s1 = F.hflip(s1)
                label = F.hflip(label.unsqueeze(0)).squeeze(0)
            if random.random() > 0.5:
                s1 = F.vflip(s1)
                label = F.vflip(label.unsqueeze(0)).squeeze(0)

        # Channel-wise normalization
        if self.normalize:
            for c in range(s1.shape[0]):
                s1[c] = (s1[c] - S1_MEAN[c]) / S1_STD[c]

        return s1, label

    @staticmethod
    def _get_crop_params(h, w, crop_size):
        if h < crop_size or w < crop_size:
            raise ValueError(f"Image ({h}x{w}) smaller than crop ({crop_size})")
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        return i, j, crop_size, crop_size


class Sen1Floods11MultiModal(Dataset):
    """Sen1Floods11 dataset loading S1 + S2 for dual-encoder fusion models.

    Returns (s1, s2, label) where s1 and s2 are separate tensors so the
    model can process them through independent encoder branches.

    Args:
        split_csv: Path to CSV file with (s1_filename, label_filename) rows.
        s1_dir: Directory containing S1 .tif files.
        s2_dir: Directory containing S2 .tif files.
        label_dir: Directory containing label .tif files.
        crop_size: Random crop size for training. None = full 512.
        augment: Whether to apply random flips.
        normalize: Whether to z-score normalize both modalities.
    """

    def __init__(
        self,
        split_csv,
        s1_dir,
        s2_dir,
        label_dir,
        crop_size=256,
        augment=False,
        normalize=True,
    ):
        self.s1_dir = Path(s1_dir)
        self.s2_dir = Path(s2_dir)
        self.label_dir = Path(label_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = normalize

        df = pd.read_csv(split_csv, header=None, names=["s1_file", "label_file"])
        self.samples = list(zip(df["s1_file"], df["label_file"]))

    def __len__(self):
        return len(self.samples)

    def _read_s1(self, path):
        """Read S1 chip, clip to [-50, 1], scale to [0, 1]."""
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, -50, 1)
        data = (data + 50) / 51.0
        return data

    def _read_s2(self, path):
        """Read S2 chip, scale by 10000, clamp negatives."""
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (13, H, W)
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, 0, 10000)
        data = data / 10000.0
        return data

    def _read_label(self, path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.int64)
        data[data == -1] = 255
        return data

    def __getitem__(self, idx):
        s1_file, label_file = self.samples[idx]

        # Derive S2 filename: replace S1Hand with S2Hand
        s2_file = s1_file.replace("S1Hand", "S2Hand")

        s1 = self._read_s1(self.s1_dir / s1_file)
        s2 = self._read_s2(self.s2_dir / s2_file)
        label = self._read_label(self.label_dir / label_file)

        s1 = torch.from_numpy(s1)      # (2, H, W)
        s2 = torch.from_numpy(s2)      # (13, H, W)
        label = torch.from_numpy(label) # (H, W)

        # Same random crop applied to all three
        if self.crop_size is not None:
            _, h, w = s1.shape
            i, j, th, tw = Sen1Floods11._get_crop_params(h, w, self.crop_size)
            s1 = s1[:, i:i + th, j:j + tw]
            s2 = s2[:, i:i + th, j:j + tw]
            label = label[i:i + th, j:j + tw]

        # Same random flips applied to all three
        if self.augment:
            if random.random() > 0.5:
                s1 = F.hflip(s1)
                s2 = F.hflip(s2)
                label = F.hflip(label.unsqueeze(0)).squeeze(0)
            if random.random() > 0.5:
                s1 = F.vflip(s1)
                s2 = F.vflip(s2)
                label = F.vflip(label.unsqueeze(0)).squeeze(0)

        # Channel-wise normalization
        if self.normalize:
            for c in range(s1.shape[0]):
                s1[c] = (s1[c] - S1_MEAN[c]) / S1_STD[c]
            for c in range(s2.shape[0]):
                s2[c] = (s2[c] - S2_MEAN[c]) / S2_STD[c]

        return s1, s2, label


def get_dataloaders(
    data_root,
    splits_dir,
    batch_size=8,
    num_workers=4,
    crop_size=256,
):
    """Create train/val/test dataloaders (S1-only).

    Args:
        data_root: Path to HandLabeled directory (contains S1Hand/, LabelHand/).
        splits_dir: Path to splits/flood_handlabeled/ directory.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        crop_size: Random crop size for training.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders.
    """
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    s1_dir = data_root / "S1Hand"
    label_dir = data_root / "LabelHand"

    train_ds = Sen1Floods11(
        split_csv=splits_dir / "flood_train_data.csv",
        s1_dir=s1_dir,
        label_dir=label_dir,
        crop_size=crop_size,
        augment=True,
        normalize=True,
    )
    val_ds = Sen1Floods11(
        split_csv=splits_dir / "flood_valid_data.csv",
        s1_dir=s1_dir,
        label_dir=label_dir,
        crop_size=None,  # full 512x512
        augment=False,
        normalize=True,
    )
    test_ds = Sen1Floods11(
        split_csv=splits_dir / "flood_test_data.csv",
        s1_dir=s1_dir,
        label_dir=label_dir,
        crop_size=None,
        augment=False,
        normalize=True,
    )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": torch.utils.data.DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    return loaders


def get_multimodal_dataloaders(
    data_root,
    splits_dir,
    batch_size=4,
    num_workers=4,
    crop_size=256,
):
    """Create train/val/test dataloaders for S1+S2 multi-modal models.

    Returns (s1, s2, label) per sample instead of (s1, label).
    """
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    s1_dir = data_root / "S1Hand"
    s2_dir = data_root / "S2Hand"
    label_dir = data_root / "LabelHand"

    train_ds = Sen1Floods11MultiModal(
        split_csv=splits_dir / "flood_train_data.csv",
        s1_dir=s1_dir, s2_dir=s2_dir, label_dir=label_dir,
        crop_size=crop_size, augment=True, normalize=True,
    )
    val_ds = Sen1Floods11MultiModal(
        split_csv=splits_dir / "flood_valid_data.csv",
        s1_dir=s1_dir, s2_dir=s2_dir, label_dir=label_dir,
        crop_size=None, augment=False, normalize=True,
    )
    test_ds = Sen1Floods11MultiModal(
        split_csv=splits_dir / "flood_test_data.csv",
        s1_dir=s1_dir, s2_dir=s2_dir, label_dir=label_dir,
        crop_size=None, augment=False, normalize=True,
    )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": torch.utils.data.DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    return loaders


class Sen1Floods11TriModal(Dataset):
    """S1 + S2 + DEM tri-modal dataset. Returns (s1, s2, dem, label)."""

    def __init__(self, split_csv, s1_dir, s2_dir, dem_dir, label_dir,
                 crop_size=256, augment=False, normalize=True):
        self.s1_dir = Path(s1_dir); self.s2_dir = Path(s2_dir)
        self.dem_dir = Path(dem_dir); self.label_dir = Path(label_dir)
        self.crop_size = crop_size; self.augment = augment; self.normalize = normalize
        df = pd.read_csv(split_csv, header=None, names=["s1_file", "label_file"])
        self.samples = list(zip(df["s1_file"], df["label_file"]))

    def __len__(self):
        return len(self.samples)

    def _read_s1(self, path):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        return (np.clip(data, -50, 1) + 50) / 51.0

    def _read_s2(self, path):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        return np.clip(np.nan_to_num(data, nan=0.0), 0, 10000) / 10000.0

    def _read_dem(self, path):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        return np.nan_to_num(data, nan=0.0)

    def _read_label(self, path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.int64)
        data[data == -1] = 255
        return data

    def __getitem__(self, idx):
        s1_file, label_file = self.samples[idx]
        s1    = torch.from_numpy(self._read_s1(self.s1_dir / s1_file))
        s2    = torch.from_numpy(self._read_s2(self.s2_dir / s1_file.replace("S1Hand", "S2Hand")))
        dem   = torch.from_numpy(self._read_dem(self.dem_dir / s1_file.replace("S1Hand", "DEMHand")))
        label = torch.from_numpy(self._read_label(self.label_dir / label_file))

        if self.crop_size is not None:
            _, h, w = s1.shape
            i, j, th, tw = Sen1Floods11._get_crop_params(h, w, self.crop_size)
            s1 = s1[:, i:i+th, j:j+tw]; s2 = s2[:, i:i+th, j:j+tw]
            dem = dem[:, i:i+th, j:j+tw]; label = label[i:i+th, j:j+tw]

        if self.augment:
            if random.random() > 0.5:
                s1 = F.hflip(s1); s2 = F.hflip(s2); dem = F.hflip(dem)
                label = F.hflip(label.unsqueeze(0)).squeeze(0)
            if random.random() > 0.5:
                s1 = F.vflip(s1); s2 = F.vflip(s2); dem = F.vflip(dem)
                label = F.vflip(label.unsqueeze(0)).squeeze(0)

        if self.normalize:
            for c in range(s1.shape[0]):
                s1[c] = (s1[c] - S1_MEAN[c]) / S1_STD[c]
            for c in range(s2.shape[0]):
                s2[c] = (s2[c] - S2_MEAN[c]) / S2_STD[c]
            for c in range(dem.shape[0]):
                dem[c] = (dem[c] - DEM_MEAN[c]) / DEM_STD[c]

        return s1, s2, dem, label


def get_trimodal_dataloaders(data_root, splits_dir, batch_size=4,
                              num_workers=4, crop_size=256):
    """Train/val/test dataloaders for S1+S2+DEM. Returns (s1, s2, dem, label)."""
    data_root = Path(data_root); splits_dir = Path(splits_dir)
    dirs = dict(s1_dir=data_root/"S1Hand", s2_dir=data_root/"S2Hand",
                dem_dir=data_root/"DEMHand", label_dir=data_root/"LabelHand")
    train_ds = Sen1Floods11TriModal(splits_dir/"flood_train_data.csv",
                                    **dirs, crop_size=crop_size, augment=True)
    val_ds   = Sen1Floods11TriModal(splits_dir/"flood_valid_data.csv",
                                    **dirs, crop_size=None, augment=False)
    test_ds  = Sen1Floods11TriModal(splits_dir/"flood_test_data.csv",
                                    **dirs, crop_size=None, augment=False)
    return {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                    shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        "val":   torch.utils.data.DataLoader(val_ds, batch_size=1,
                    shuffle=False, num_workers=num_workers, pin_memory=True),
        "test":  torch.utils.data.DataLoader(test_ds, batch_size=1,
                    shuffle=False, num_workers=num_workers, pin_memory=True),
    }
