"""Dual-Encoder U-Net with Cross-Attention Fusion for S1+S2 flood segmentation.

Architecture:
    S1 (2ch) → S1 Encoder (ResNet34) → S1 features at 4 scales
    S2 (13ch) → S2 Encoder (ResNet34) → S2 features at 4 scales
                                          ↓
                        Cross-Attention Fusion at each scale
                                          ↓
                              U-Net Decoder → (B, 2, H, W)

Cross-attention lets each modality attend to the other, learning which
spatial regions of S2 (optical) are informative for disambiguating S1 (SAR)
features and vice versa. This is the key novelty over early fusion (concatenation).

Memory-optimized for RTX 5070 8GB:
- ResNet34 backbone (~21M params per encoder, vs ~25M for ResNet50)
- Lightweight cross-attention with reduced projection dimension
- GroupNorm instead of BatchNorm (works with small batch sizes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torchvision.models import resnet34


def convert_bn_to_gn(module, num_groups=16):
    """Recursively replace BatchNorm2d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            nc = child.num_features
            ng = min(num_groups, nc)
            while nc % ng != 0:
                ng -= 1
            setattr(module, name, nn.GroupNorm(ng, nc, affine=True))
        else:
            convert_bn_to_gn(child, num_groups)
    return module


class ResNet34Encoder(nn.Module):
    """ResNet34 encoder that returns features at 4 scales.

    Output channels at each scale:
        scale 0: 64   (1/4 resolution)
        scale 1: 128  (1/8 resolution)
        scale 2: 256  (1/16 resolution)
        scale 3: 512  (1/32 resolution)
    """

    def __init__(self, in_channels):
        super().__init__()
        base = resnet34(weights=None)
        convert_bn_to_gn(base)

        # Replace first conv for custom input channels
        self.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.GroupNorm(16, 64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = base.layer1  # 64 ch, 1/4
        self.layer2 = base.layer2  # 128 ch, 1/8
        self.layer3 = base.layer3  # 256 ch, 1/16
        self.layer4 = base.layer4  # 512 ch, 1/32

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)

        return [f1, f2, f3, f4]


class CrossAttention(nn.Module):
    """Bi-directional cross-attention between two feature maps.

    Given features A and B of the same spatial size:
        A_out = A + Attn(Q=A, K=B, V=B)
        B_out = B + Attn(Q=B, K=A, V=A)

    Uses a reduced projection dimension to save memory.
    """

    def __init__(self, channels, num_heads=4, proj_ratio=0.25):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.proj_dim = max(num_heads, int(channels * proj_ratio))
        # Round to multiple of num_heads
        self.proj_dim = (self.proj_dim // num_heads) * num_heads
        self.head_dim = self.proj_dim // num_heads

        # Projections for A→B attention
        self.q_a = nn.Conv2d(channels, self.proj_dim, 1)
        self.k_b = nn.Conv2d(channels, self.proj_dim, 1)
        self.v_b = nn.Conv2d(channels, self.proj_dim, 1)
        self.out_a = nn.Conv2d(self.proj_dim, channels, 1)

        # Projections for B→A attention
        self.q_b = nn.Conv2d(channels, self.proj_dim, 1)
        self.k_a = nn.Conv2d(channels, self.proj_dim, 1)
        self.v_a = nn.Conv2d(channels, self.proj_dim, 1)
        self.out_b = nn.Conv2d(self.proj_dim, channels, 1)

        self.norm_a = nn.GroupNorm(min(16, channels), channels)
        self.norm_b = nn.GroupNorm(min(16, channels), channels)
        self.scale = self.head_dim ** -0.5

    def _attend(self, q, k, v, out_proj):
        """Attention in float32 to prevent fp16 overflow in softmax."""
        B, _, H, W = q.shape
        # Force float32 for numerical stability regardless of AMP context
        q = q.float().view(B, self.num_heads, self.head_dim, H * W)
        k = k.float().view(B, self.num_heads, self.head_dim, H * W)
        v = v.float().view(B, self.num_heads, self.head_dim, H * W)

        # Attention: (B, heads, H*W, H*W)
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = torch.matmul(v, attn.transpose(-2, -1))  # (B, heads, head_dim, H*W)
        out = out.view(B, self.proj_dim, H, W)
        return out_proj(out)

    def forward(self, feat_a, feat_b):
        # A attends to B
        a_out = feat_a + self._attend(
            self.q_a(feat_a), self.k_b(feat_b), self.v_b(feat_b), self.out_a
        )
        a_out = self.norm_a(a_out)

        # B attends to A
        b_out = feat_b + self._attend(
            self.q_b(feat_b), self.k_a(feat_a), self.v_a(feat_a), self.out_b
        )
        b_out = self.norm_b(b_out)

        return a_out, b_out


class DecoderBlock(nn.Module):
    """U-Net decoder block: upsample + concat skip + 2x conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        total_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd-sized inputs
        if x.shape != skip.shape:
            x = TF.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FusionUNet(nn.Module):
    """Dual-encoder U-Net with cross-attention fusion.

    Two independent ResNet34 encoders process S1 and S2, cross-attention
    modules fuse information at each encoder scale, and a shared U-Net
    decoder produces the final segmentation.

    Args:
        s1_channels: Number of S1 input channels (default: 2 for VV/VH).
        s2_channels: Number of S2 input channels (default: 13 bands).
        num_classes: Output classes (default: 2 for water/non-water).
        attention_heads: Number of attention heads per cross-attention module.
        dropout_rate: Dropout rate for MC Dropout uncertainty (Phase 3).
    """

    # Encoder channel counts at each scale
    ENC_CHANNELS = [64, 128, 256, 512]

    def __init__(
        self,
        s1_channels=2,
        s2_channels=13,
        num_classes=2,
        attention_heads=4,
        dropout_rate=0.1,
    ):
        super().__init__()

        # Dual encoders
        self.s1_encoder = ResNet34Encoder(s1_channels)
        self.s2_encoder = ResNet34Encoder(s2_channels)

        # Cross-attention at each encoder scale
        self.cross_attn = nn.ModuleList([
            CrossAttention(ch, num_heads=min(attention_heads, ch // 16))
            for ch in self.ENC_CHANNELS
        ])

        # Fusion: concat S1+S2 features → project to single feature map
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch, 1, bias=False),
                nn.GroupNorm(min(16, ch), ch),
                nn.ReLU(inplace=True),
            )
            for ch in self.ENC_CHANNELS
        ])

        # Decoder: bottom-up path
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)

        # Final upsample from 1/4 to full resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, s1, s2):
        """Forward pass.

        Args:
            s1: (B, 2, H, W) S1 SAR input.
            s2: (B, 13, H, W) S2 optical input.

        Returns:
            (B, num_classes, H, W) logits.
        """
        # Encode both modalities
        s1_feats = self.s1_encoder(s1)  # list of 4 feature maps
        s2_feats = self.s2_encoder(s2)

        # Cross-attention fusion at each scale
        fused = []
        for s1_f, s2_f, attn, fuse in zip(
            s1_feats, s2_feats, self.cross_attn, self.fuse_convs
        ):
            s1_att, s2_att = attn(s1_f, s2_f)
            merged = torch.cat([s1_att, s2_att], dim=1)
            fused.append(fuse(merged))

        # Decoder with skip connections
        # fused: [1/4, 1/8, 1/16, 1/32]
        x = self.decoder4(fused[3], fused[2])  # 1/32 → 1/16
        x = self.decoder3(x, fused[1])          # 1/16 → 1/8
        x = self.decoder2(x, fused[0])          # 1/8 → 1/4

        x = self.final_up(x)                    # 1/4 → full
        x = self.dropout(x)
        x = self.head(x)

        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = FusionUNet(s1_channels=2, s2_channels=13, num_classes=2)
    print(f"Parameters: {count_parameters(model):,}")

    s1 = torch.randn(2, 2, 256, 256)
    s2 = torch.randn(2, 13, 256, 256)
    out = model(s1, s2)
    print(f"Input S1: {s1.shape}, S2: {s2.shape}")
    print(f"Output: {out.shape}")
