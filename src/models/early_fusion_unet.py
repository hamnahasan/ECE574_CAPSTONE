"""Early Fusion U-Net baseline for multi-modal flood segmentation.

PURPOSE (scientific):
    This model serves as the critical ablation baseline that justifies the
    complexity of our cross-attention dual/tri-encoder architecture.

    Early fusion = concatenate all modalities into a single tensor BEFORE
    the encoder. The encoder then sees all 17 channels (2 S1 + 13 S2 + 2 DEM)
    as if they were one image.

    WHY THIS IS THE BASELINE TO BEAT:
    - It is the simplest possible multi-modal fusion strategy
    - It has been used in prior work (Bai et al. 2021, BASNet)
    - If our cross-attention model does NOT outperform this, the added
      complexity of separate encoders is not justified

    EXPECTED WEAKNESS of early fusion:
    - Forces the encoder to learn joint S1+S2+DEM representations from
      the very first layer, which is difficult because the modalities have
      fundamentally different statistical properties (SAR dB vs optical
      reflectance vs elevation in meters)
    - Cannot selectively attend to one modality based on spatial context
      (e.g., trust SAR over optical in cloudy regions)
    - All modalities are treated equally regardless of their reliability

Architecture:
    Cat(S1[2ch], S2[13ch], DEM[2ch]) = 17ch input
    -> ResNet34 encoder (modified first conv: 3->17 channels)
    -> Standard U-Net decoder
    -> (B, 2, H, W) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torchvision.models import resnet34

from src.models.fusion_unet import convert_bn_to_gn, DecoderBlock, count_parameters


class EarlyFusionUNet(nn.Module):
    """Single-encoder U-Net with early (input-level) fusion of all modalities.

    All modalities are concatenated at the input and processed by one shared
    ResNet34 encoder. This is the simplest possible fusion strategy and serves
    as the upper-bound baseline for single-encoder approaches.

    Args:
        in_channels: Total input channels after concatenation.
                     Default 17 = S1(2) + S2(13) + DEM(2).
                     Use 15 for S1+S2 only, 4 for S1+DEM only, etc.
        num_classes: Output classes (2 for water/non-water).
        dropout_rate: Spatial dropout for MC uncertainty estimation.
    """

    def __init__(self, in_channels=17, num_classes=2, dropout_rate=0.1):
        super().__init__()

        # Load ResNet34 and replace BatchNorm with GroupNorm.
        # GroupNorm is used throughout this project because it is stable
        # at small batch sizes (batch_size=4 with 8GB VRAM).
        base = resnet34(weights=None)
        convert_bn_to_gn(base)

        # Replace first conv: ImageNet expects 3 channels, we have in_channels.
        # We do NOT use pretrained weights because SAR/DEM inputs have no
        # correspondence to RGB statistics.
        self.conv1   = nn.Conv2d(in_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.bn1     = nn.GroupNorm(16, 64, affine=True)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet34 residual blocks — shared across all modalities.
        # Skip connections are taken from each layer for the decoder.
        self.layer1 = base.layer1  # 64ch,  H/4
        self.layer2 = base.layer2  # 128ch, H/8
        self.layer3 = base.layer3  # 256ch, H/16
        self.layer4 = base.layer4  # 512ch, H/32

        # U-Net decoder: progressively upsample and fuse skip connections.
        # Skip connections carry fine-grained spatial detail from the encoder
        # back to the decoder, which is critical for sharp flood boundaries.
        self.decoder4 = DecoderBlock(512, 256, 256)  # 1/32 -> 1/16
        self.decoder3 = DecoderBlock(256, 128, 128)  # 1/16 -> 1/8
        self.decoder2 = DecoderBlock(128,  64,  64)  # 1/8  -> 1/4

        # Final 4x upsample to restore full input resolution.
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # Spatial dropout: randomly zero entire feature map channels.
        # At inference we keep dropout ON for MC uncertainty estimation (Phase 4).
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # 1x1 conv: project 32 feature channels to num_classes logits.
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, in_channels, H, W) — pre-concatenated modalities.
               Caller is responsible for concatenation: Cat(s1, s2, dem).

        Returns:
            (B, num_classes, H, W) logits.
        """
        # Encoder — extract features at 4 scales
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = self.relu(x)
        x  = self.maxpool(x)

        f1 = self.layer1(x)   # (B, 64,  H/4,  W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8,  W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)

        # Decoder — upsample with skip connections
        x = self.decoder4(f4, f3)
        x = self.decoder3(x,  f2)
        x = self.decoder2(x,  f1)

        x = self.final_up(x)
        x = self.dropout(x)
        return self.head(x)


def build_early_fusion(modalities="s1_s2_dem", num_classes=2, dropout_rate=0.1):
    """Convenience factory — builds EarlyFusionUNet for a given modality set.

    Args:
        modalities: One of 's1_s2_dem' (17ch), 's1_s2' (15ch),
                    's1_dem' (4ch), 's2_dem' (15ch), 's1' (2ch).
        num_classes: Output classes.
        dropout_rate: Spatial dropout probability.

    Returns:
        (model, in_channels) tuple.
    """
    # Channel counts per modality
    ch = {"s1": 2, "s2": 13, "dem": 2}
    total = sum(ch[m] for m in modalities.split("_"))
    return EarlyFusionUNet(in_channels=total, num_classes=num_classes,
                           dropout_rate=dropout_rate), total


if __name__ == "__main__":
    model, c = build_early_fusion("s1_s2_dem")
    print(f"Early Fusion U-Net ({c}ch input): {count_parameters(model):,} parameters")

    x = torch.randn(2, c, 256, 256)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
