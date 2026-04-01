"""FCN-ResNet50 baseline for S1-only flood segmentation.

Replicates the model from the Sen1Floods11 paper (Bonafilia et al., 2020):
- torchvision fcn_resnet50 with 2-channel input (VV, VH)
- BatchNorm → GroupNorm conversion
- 2-class output (non-water, water)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import FCN_ResNet50_Weights


def convert_bn_to_gn(module, num_groups=32):
    """Recursively replace BatchNorm2d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # GroupNorm requires num_groups to divide num_channels
            ng = min(num_groups, num_channels)
            while num_channels % ng != 0:
                ng -= 1
            gn = nn.GroupNorm(ng, num_channels, affine=True)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)
    return module


def build_fcn_baseline(in_channels=2, num_classes=2, pretrained_backbone=False):
    """Build FCN-ResNet50 model matching the Sen1Floods11 baseline.

    Args:
        in_channels: Number of input channels (2 for S1 VV/VH).
        num_classes: Number of output classes (2 for water/non-water).
        pretrained_backbone: Whether to use ImageNet-pretrained ResNet50.

    Returns:
        nn.Module: FCN-ResNet50 model.
    """
    if pretrained_backbone:
        net = models.segmentation.fcn_resnet50(
            weights=None,
            num_classes=num_classes,
            weights_backbone=models.ResNet50_Weights.DEFAULT,
        )
    else:
        net = models.segmentation.fcn_resnet50(
            weights=None,
            num_classes=num_classes,
            weights_backbone=None,
        )

    # Replace first conv: 3 channels → in_channels
    old_conv = net.backbone.conv1
    net.backbone.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # BatchNorm → GroupNorm (matches original paper)
    convert_bn_to_gn(net)

    return net


class FCNBaseline(nn.Module):
    """Wrapper around FCN-ResNet50 for cleaner forward pass."""

    def __init__(self, in_channels=2, num_classes=2, pretrained_backbone=False):
        super().__init__()
        self.net = build_fcn_baseline(in_channels, num_classes, pretrained_backbone)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, num_classes, H, W) logits.
        """
        out = self.net(x)
        return out["out"]
