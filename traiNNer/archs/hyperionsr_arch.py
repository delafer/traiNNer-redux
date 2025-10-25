#!/usr/bin/env python3
"""
HyperionSR: A State-of-the-Art Super-Resolution and Restoration Network.

Author: Philip Hofmann

Description:
HyperionSR is an advanced super-resolution architecture that integrates modern
deep learning concepts to achieve a superior balance of performance, quality, and
computational efficiency. It is designed to be a powerful, drop-in replacement
for other architectures within the traiNNer-redux ecosystem and is suitable
for ONNX export.

Core Innovations:
- HyperionBlock: The core of the network, featuring a Transformer-style Gated
  Feed-Forward Network (Gated-FFN) and GroupNorm for stable, powerful feature transformation.
- Dual-Attention Mechanism: Combines a lightweight Spatial Attention Gate with
  Channel Attention in every block to model both "where" and "what" to focus on.
- Residual Group Hierarchy: Organizes blocks into groups with multi-level residual
  connections, allowing for the construction of extremely deep and stable networks.
- Hierarchical Feature Fusion: Employs a dedicated fusion module with a long
  skip connection to intelligently combine shallow and deep features.

Usage:
- Place this file in the `traiNNer/archs/` directory.
- Use in your config.yaml, e.g.: `architecture: HyperionSR_M` (Note the class name)
"""

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

# --- Re-usable Building Blocks ---


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, num_feat: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // reduction, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // reduction, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv_du(self.avg_pool(x))


class SpatialAttentionGate(nn.Module):
    """Lightweight Spatial Attention Gate to model 'where' to focus."""

    def __init__(self, num_feat: int) -> None:
        super().__init__()
        self.spatial_gate = nn.Conv2d(num_feat, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.spatial_gate(x))


class GatedFFN(nn.Module):
    """Gated Feed-Forward Network inspired by modern Transformers."""

    def __init__(self, num_feat: int, ffn_expansion: int = 2) -> None:
        super().__init__()
        hidden_features = num_feat * ffn_expansion
        self.project_in = nn.Conv2d(num_feat, hidden_features * 2, 1)
        self.project_out = nn.Conv2d(hidden_features, num_feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.project_in(x)
        x1, x2 = x_proj.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


# --- The Core Hyperion Block and Group ---


class HyperionBlock(nn.Module):
    """The core block of HyperionSR, combining dual-attention and a Gated-FFN."""

    def __init__(self, num_feat: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=num_feat)
        self.attn = SpatialAttentionGate(num_feat)
        self.ca = ChannelAttention(num_feat)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=num_feat)
        self.ffn = GatedFFN(num_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.ca(x)
        x = x + x_res
        x_res = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + x_res


class ResidualGroup(nn.Module):
    """A group of HyperionBlocks with a local residual connection."""

    def __init__(self, num_feat: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[HyperionBlock(num_feat) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.blocks(x)


# --- Factory Registration for traiNNer-redux ---
# Similar to other traiNNer archs, we register wrapper classes for each variant.


@ARCH_REGISTRY.register()
class HyperionSR_S(nn.Module):
    """HyperionSR-S: For GPUs with ~6GB VRAM (e.g., GTX 1660S)."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = HyperionSR(scale=scale, num_feat=48, num_groups=3, num_blocks=3)

    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class HyperionSR_M(nn.Module):
    """HyperionSR-M: For GPUs with ~12GB VRAM (e.g., RTX 3060)."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = HyperionSR(scale=scale, num_feat=64, num_groups=4, num_blocks=4)

    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class HyperionSR_L(nn.Module):
    """HyperionSR-L: For high-end GPUs with ~24GB VRAM (e.g., RTX 4090)."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = HyperionSR(scale=scale, num_feat=128, num_groups=5, num_blocks=5)

    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class HyperionSR_XL(nn.Module):
    """HyperionSR-XL: For flagship GPUs with >24GB VRAM, pushing quality limits."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = HyperionSR(scale=scale, num_feat=160, num_groups=6, num_blocks=6)

    def forward(self, x):
        return self.model(x)


# --- The Main HyperionSR Network ---
class HyperionSR(nn.Module):
    """The underlying HyperionSR architecture. Not registered directly."""

    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 4,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualGroup(num_feat, num_blocks) for _ in range(num_groups)]
        )
        self.fusion = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow_features = self.conv_in(x)
        deep_features = self.body(shallow_features)
        fused_features = self.fusion(deep_features) + shallow_features
        out = self.upsampler(fused_features)
        out = self.conv_out(out)
        return out
