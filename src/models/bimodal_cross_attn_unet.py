"""Generic two-encoder cross-attention U-Net.

This is the architectural counterpart to the early-fusion 2-modality
ablation variants (s1_s2, s1_dem, s2_dem). Where EarlyFusionUNet
concatenates the two modality tensors at the input and processes them
through a single shared encoder, BimodalCrossAttnUNet keeps the two
modalities in independent encoder branches and fuses their feature maps
through bi-directional cross-attention at four decoder scales.

This closes the cross-attention ablation matrix that was previously
incomplete — we already have the 2-way s1+s2 (FusionUNet) and the 3-way
s1+s2+dem (TriModalFusionUNet); this class adds s1+dem and s2+dem so the
matrix has all four cells.

The cross-attention block, decoder block, and ResNet34 encoder are reused
from src.models.fusion_unet so the fusion pathway is byte-for-byte
identical to the existing FusionUNet — only the modality identity differs.

Usage:
    model = BimodalCrossAttnUNet(a_channels=2, b_channels=2)   # S1 + DEM
    model = BimodalCrossAttnUNet(a_channels=13, b_channels=2)  # S2 + DEM
    out   = model(a, b)
"""

import torch
import torch.nn as nn

from src.models.fusion_unet import (
    ResNet34Encoder,
    CrossAttention,
    DecoderBlock,
    count_parameters,
)


class BimodalCrossAttnUNet(nn.Module):
    """Two-encoder U-Net with bi-directional cross-attention fusion.

    Args:
        a_channels:      Input channels for modality A (e.g. 2 for S1).
        b_channels:      Input channels for modality B (e.g. 2 for DEM).
        num_classes:     Output classes (default 2: water / non-water).
        attention_heads: Heads per cross-attention module.
        dropout_rate:    Decoder dropout rate (used by MC-Dropout at inference).

    The generic naming (a / b) makes the model agnostic to which two
    modalities it is fusing — the caller is responsible for feeding the
    right pair of tensors. This keeps a single class usable for
    s1+s2, s1+dem, s2+dem, etc.
    """

    ENC_CHANNELS = [64, 128, 256, 512]

    def __init__(self, a_channels, b_channels, num_classes=2,
                 attention_heads=4, dropout_rate=0.1):
        super().__init__()

        # Two independent encoders — same architecture, different in_channels.
        self.a_encoder = ResNet34Encoder(a_channels)
        self.b_encoder = ResNet34Encoder(b_channels)

        # Bi-directional cross-attention at each encoder scale.
        # Heads scale with channel width so we never request more heads
        # than the channel count can support.
        self.bi_attn = nn.ModuleList([
            CrossAttention(ch, num_heads=min(attention_heads, max(1, ch // 16)))
            for ch in self.ENC_CHANNELS
        ])

        # After attention each scale yields two feature maps of `ch` channels;
        # we concatenate them and project back to `ch` so the decoder sees
        # the same channel counts as in the trimodal model.
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch, 1, bias=False),
                nn.GroupNorm(min(16, ch), ch),
                nn.ReLU(inplace=True),
            )
            for ch in self.ENC_CHANNELS
        ])

        # Decoder is identical to FusionUNet / TriModalFusionUNet so this
        # class does not introduce a confound on the decoder side.
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128,  64,  64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.head    = nn.Conv2d(32, num_classes, 1)

    def forward(self, a, b):
        """Forward pass.

        Args:
            a: (B, a_channels, H, W)
            b: (B, b_channels, H, W)

        Returns:
            (B, num_classes, H, W) logits.
        """
        a_feats = self.a_encoder(a)
        b_feats = self.b_encoder(b)

        fused = []
        for a_f, b_f, attn, fuse in zip(a_feats, b_feats, self.bi_attn, self.fuse_convs):
            a_att, b_att = attn(a_f, b_f)
            fused.append(fuse(torch.cat([a_att, b_att], dim=1)))

        x = self.decoder4(fused[3], fused[2])
        x = self.decoder3(x,        fused[1])
        x = self.decoder2(x,        fused[0])
        x = self.final_up(x)
        x = self.dropout(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Convenience constructor for the standard modality-pair channel counts.
# ---------------------------------------------------------------------------

MODALITY_CHANNELS = {"s1": 2, "s2": 13, "dem": 2}


def build_bimodal(modalities, num_classes=2, attention_heads=4, dropout_rate=0.1):
    """Build a BimodalCrossAttnUNet for a named modality pair.

    Args:
        modalities: A pair, e.g. ("s1", "dem"), ("s2", "dem"), ("s1", "s2").
                    Order matters — the first modality is fed as `a`,
                    second as `b`.

    Returns:
        BimodalCrossAttnUNet ready for training.
    """
    if len(modalities) != 2:
        raise ValueError(f"Expected pair of modalities, got {modalities}")
    a, b = modalities
    if a not in MODALITY_CHANNELS or b not in MODALITY_CHANNELS:
        raise ValueError(
            f"Unknown modality in {modalities}; "
            f"valid keys: {list(MODALITY_CHANNELS)}"
        )
    return BimodalCrossAttnUNet(
        a_channels=MODALITY_CHANNELS[a],
        b_channels=MODALITY_CHANNELS[b],
        num_classes=num_classes,
        attention_heads=attention_heads,
        dropout_rate=dropout_rate,
    )
