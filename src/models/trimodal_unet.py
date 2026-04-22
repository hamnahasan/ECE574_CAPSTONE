"""Tri-Modal U-Net with 3-way Cross-Attention Fusion (S1 + S2 + DEM).

Extends the dual-encoder fusion model (Phase 2) with a third DEM encoder.
Cross-attention is extended to 3-way: each modality attends to both others.

Architecture:
    S1  (2ch)  -> ResNet34 Encoder -> features at 4 scales
    S2  (13ch) -> ResNet34 Encoder -> features at 4 scales
    DEM (2ch)  -> ResNet34 Encoder -> features at 4 scales
                                        |
              TriModal CrossAttention at each scale
                                        |
                          U-Net Decoder -> (B, 2, H, W)

3-way attention at each scale:
    S1  attends to (S2, DEM)  -> S1_out
    S2  attends to (S1, DEM)  -> S2_out
    DEM attends to (S1, S2)   -> DEM_out
    Fuse: Conv1x1(Cat(S1_out, S2_out, DEM_out)) -> single feature map
"""

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torchvision.models import resnet34

# Re-use shared components from fusion_unet
from src.models.fusion_unet import (
    convert_bn_to_gn,
    ResNet34Encoder,
    DecoderBlock,
    count_parameters,
)


class TriModalCrossAttention(nn.Module):
    """3-way cross-attention: each modality attends to the concatenation of both others.

    Given features A, B, C of the same spatial size and channel count:
        A_out = A + Attn(Q=A, K=Cat(B,C), V=Cat(B,C))
        B_out = B + Attn(Q=B, K=Cat(A,C), V=Cat(A,C))
        C_out = C + Attn(Q=C, K=Cat(A,B), V=Cat(A,B))

    All computations forced to float32 to prevent fp16 overflow in softmax.
    """

    def __init__(self, channels, num_heads=4, proj_ratio=0.25):
        super().__init__()
        self.channels  = channels
        self.num_heads = num_heads
        self.proj_dim  = max(num_heads, int(channels * proj_ratio))
        self.proj_dim  = (self.proj_dim // num_heads) * num_heads
        self.head_dim  = self.proj_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Each modality has its own Q projection (channels -> proj_dim)
        # K and V projections receive 2*channels (concatenated other two)
        self.q_a = nn.Conv2d(channels,   self.proj_dim, 1)
        self.k_a = nn.Conv2d(channels*2, self.proj_dim, 1)
        self.v_a = nn.Conv2d(channels*2, self.proj_dim, 1)
        self.o_a = nn.Conv2d(self.proj_dim, channels, 1)

        self.q_b = nn.Conv2d(channels,   self.proj_dim, 1)
        self.k_b = nn.Conv2d(channels*2, self.proj_dim, 1)
        self.v_b = nn.Conv2d(channels*2, self.proj_dim, 1)
        self.o_b = nn.Conv2d(self.proj_dim, channels, 1)

        self.q_c = nn.Conv2d(channels,   self.proj_dim, 1)
        self.k_c = nn.Conv2d(channels*2, self.proj_dim, 1)
        self.v_c = nn.Conv2d(channels*2, self.proj_dim, 1)
        self.o_c = nn.Conv2d(self.proj_dim, channels, 1)

        self.norm_a = nn.GroupNorm(min(16, channels), channels)
        self.norm_b = nn.GroupNorm(min(16, channels), channels)
        self.norm_c = nn.GroupNorm(min(16, channels), channels)

    def _attend(self, q_proj, kv_proj_k, kv_proj_v, query, context, out_proj):
        """query: (B, C, H, W), context: (B, 2C, H, W)."""
        B, _, H, W = query.shape
        q = q_proj(query).float().view(B, self.num_heads, self.head_dim, H*W)
        k = kv_proj_k(context).float().view(B, self.num_heads, self.head_dim, H*W)
        v = kv_proj_v(context).float().view(B, self.num_heads, self.head_dim, H*W)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(v, attn.transpose(-2, -1))
        out  = out.view(B, self.proj_dim, H, W)
        return out_proj(out)

    def forward(self, feat_a, feat_b, feat_c):
        # A attends to B+C
        bc = torch.cat([feat_b, feat_c], dim=1)
        a_out = self.norm_a(feat_a + self._attend(self.q_a, self.k_a, self.v_a,
                                                   feat_a, bc, self.o_a))
        # B attends to A+C
        ac = torch.cat([feat_a, feat_c], dim=1)
        b_out = self.norm_b(feat_b + self._attend(self.q_b, self.k_b, self.v_b,
                                                   feat_b, ac, self.o_b))
        # C attends to A+B
        ab = torch.cat([feat_a, feat_b], dim=1)
        c_out = self.norm_c(feat_c + self._attend(self.q_c, self.k_c, self.v_c,
                                                   feat_c, ab, self.o_c))
        return a_out, b_out, c_out


class TriModalFusionUNet(nn.Module):
    """Tri-encoder U-Net with 3-way cross-attention fusion (S1 + S2 + DEM).

    Args:
        s1_channels:  S1 input channels (default 2: VV, VH).
        s2_channels:  S2 input channels (default 13 bands).
        dem_channels: DEM input channels (default 2: elevation, slope).
        num_classes:  Output classes (default 2: water / non-water).
        attention_heads: Heads per cross-attention module.
        dropout_rate: Dropout for MC uncertainty (Phase 4).
    """

    ENC_CHANNELS = [64, 128, 256, 512]

    def __init__(self, s1_channels=2, s2_channels=13, dem_channels=2,
                 num_classes=2, attention_heads=4, dropout_rate=0.1):
        super().__init__()

        # Three independent encoders
        self.s1_encoder  = ResNet34Encoder(s1_channels)
        self.s2_encoder  = ResNet34Encoder(s2_channels)
        self.dem_encoder = ResNet34Encoder(dem_channels)

        # 3-way cross-attention at each encoder scale
        self.tri_attn = nn.ModuleList([
            TriModalCrossAttention(ch, num_heads=min(attention_heads, max(1, ch//16)))
            for ch in self.ENC_CHANNELS
        ])

        # Fuse 3 attended feature maps -> 1 via Conv1x1
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 3, ch, 1, bias=False),
                nn.GroupNorm(min(16, ch), ch),
                nn.ReLU(inplace=True),
            )
            for ch in self.ENC_CHANNELS
        ])

        # Decoder (identical to FusionUNet)
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

    def forward(self, s1, s2, dem):
        """Forward pass.

        Args:
            s1:  (B, 2,  H, W)
            s2:  (B, 13, H, W)
            dem: (B, 2,  H, W)

        Returns:
            (B, num_classes, H, W) logits.
        """
        s1_feats  = self.s1_encoder(s1)   # 4 feature maps
        s2_feats  = self.s2_encoder(s2)
        dem_feats = self.dem_encoder(dem)

        fused = []
        for s1_f, s2_f, dem_f, attn, fuse in zip(
            s1_feats, s2_feats, dem_feats, self.tri_attn, self.fuse_convs
        ):
            s1_att, s2_att, dem_att = attn(s1_f, s2_f, dem_f)
            fused.append(fuse(torch.cat([s1_att, s2_att, dem_att], dim=1)))

        # fused: [1/4, 1/8, 1/16, 1/32]
        x = self.decoder4(fused[3], fused[2])
        x = self.decoder3(x,        fused[1])
        x = self.decoder2(x,        fused[0])
        x = self.final_up(x)
        x = self.dropout(x)
        return self.head(x)
