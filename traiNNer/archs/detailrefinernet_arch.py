import torch
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY


# Squeeze-and-Excitation Layer for Channel Attention
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16) -> None:
        super().__init__()
        # Global average pooling to "squeeze" spatial information
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # "Excitation" layers to learn channel weights
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),  # GELU is a smooth alternative to ReLU
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Reweight the input feature map
        return x * y.expand_as(x)


# Enhanced Refinement Block with Channel Attention
class EnhancedRefinementBlock(nn.Module):
    def __init__(self, num_features=64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.se = SELayer(num_features)

    def forward(self, x):
        # Local residual connection
        res = self.conv2(self.act(self.conv1(x)))
        res = self.se(res)
        return x + res


@ARCH_REGISTRY.register()
class DetailRefinerNet(nn.Module):
    """
    An enhanced 1x restoration network with channel attention and
    long-range feature fusion, suitable for tasks like artifact removal,
    sharpening, and refinement. Designed to be efficient for an RTX 3060.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        num_features: int = 64,
        num_groups: int = 4,
        num_blocks_per_group: int = 4,
    ) -> None:
        super().__init__()

        # 1. Shallow Feature Extraction
        self.initial_conv = nn.Conv2d(in_ch, num_features, 3, 1, 1)

        # 2. Deep Feature Extraction with groups of ERBs
        self.groups = nn.ModuleList()
        for _ in range(num_groups):
            self.groups.append(
                nn.Sequential(
                    *[
                        EnhancedRefinementBlock(num_features)
                        for _ in range(num_blocks_per_group)
                    ]
                )
            )

        # 3. Long-Range Feature Fusion
        # This layer fuses the outputs from all groups
        self.fusion_conv = nn.Conv2d(num_features * num_groups, num_features, 1, 1, 0)

        # 4. Final Reconstruction Layer
        self.final_conv = nn.Conv2d(num_features, out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original input for the final global residual connection
        shortcut = x

        # Extract shallow features
        x = self.initial_conv(x)

        # Process through groups and collect outputs for fusion
        group_outputs = []
        for group in self.groups:
            x = group(x)
            group_outputs.append(x)

        # Fuse the features from all groups
        fused_features = self.fusion_conv(torch.cat(group_outputs, dim=1))

        # Reconstruct the residual image
        residual = self.final_conv(fused_features)

        # Add the residual to the original input
        return shortcut + residual
