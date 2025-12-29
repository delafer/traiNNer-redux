#!/usr/bin/env python3
"""
ElysiumSR: A High-Performance, General-Purpose Super-Resolution Network.

Author: Philip Hofmann

Description:
ElysiumSR is a convolutional neural network designed for high-quality image
super-resolution and restoration. It is built upon a foundation of proven
Residual Blocks, making it both efficient and effective. The architecture is
designed to be easily integrated into the traiNNer-redux framework and is
optimized for ONNX export.

Core Design Philosophy:
- Simplicity and Efficiency: Utilizes a classic and robust residual architecture
  that is computationally efficient, allowing for fast training and inference.
- Scalability: The network is designed in several variants to cater to a wide
  range of hardware, from mid-range GPUs with 6GB VRAM to high-end consumer
  cards with 24GB+ VRAM.
- Quality Focus: While efficient, the design choices, such as stochastic depth
  (DropPath) and a long skip connection for feature fusion, are made to
  maximize image quality and restoration capability.

Variants:
- ElysiumSR-S: Small model, designed for GPUs with ~6GB VRAM (e.g., GTX 1660 Super).
- ElysiumSR-M: Medium model, balanced for GPUs with ~12GB VRAM (e.g., RTX 3060).
- ElysiumSR-L: Large model, for high-end GPUs with ~24GB VRAM (e.g., RTX 4090).
- ElysiumSR-XL: Extra-Large model, for flagship GPUs with >24GB VRAM.

Usage:
- Place this file in the `traiNNer/archs/` directory.
- Use in your config.yaml, e.g.: `architecture: ElysiumSR_S` (Note the class name)
"""

from typing import Literal

import torch
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

# --- Building Blocks ---


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """
    Drop paths (Stochastic Depth) per sample.
    This function is from the timm library.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 4D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main module).
    This class is from the timm library.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob:.3f}"


class ResidualBlock(nn.Module):
    """
    A fundamental Residual Block consisting of two convolutional layers with a
    ReLU activation in between. The input is added back to the output of the
    second convolution, allowing the network to learn residual mappings.
    """

    def __init__(self, num_feat: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


# --- Main ElysiumSR Network ---

# Note: traiNNer-redux registers the class name directly.
# The factory functions below are for convenience and configuration,
# but the registry will see the class name (e.g., ElysiumSR_M).


@ARCH_REGISTRY.register()
class ElysiumSR_S(nn.Module):
    """ElysiumSR-S: For GPUs with ~6GB VRAM (e.g., GTX 1660S)."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = ElysiumSR(
            scale=scale, num_feat=64, num_blocks=10, drop_path_rate=0.0
        )

    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class ElysiumSR_M(nn.Module):
    """ElysiumSR-M: Balanced for GPUs with ~12GB VRAM (e.g., RTX 3060)."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = ElysiumSR(
            scale=scale, num_feat=80, num_blocks=16, drop_path_rate=0.05
        )

    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class ElysiumSR_L(nn.Module):
    """ElysiumSR-L: For high-end GPUs with ~24GB VRAM (e.g., RTX 4090)."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = ElysiumSR(
            scale=scale, num_feat=128, num_blocks=24, drop_path_rate=0.1
        )

    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class ElysiumSR_XL(nn.Module):
    """ElysiumSR-XL: For flagship GPUs with >24GB VRAM."""

    def __init__(self, scale: int = 4, **kwargs) -> None:
        super().__init__()
        self.model = ElysiumSR(
            scale=scale, num_feat=160, num_blocks=32, drop_path_rate=0.1
        )

    def forward(self, x):
        return self.model(x)


class ElysiumSR(nn.Module):
    """The underlying ElysiumSR architecture. Not registered directly."""

    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 80,
        num_blocks: int = 12,
        drop_path_rate: float = 0.0,
        upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    ) -> None:
        super().__init__()
        self.scale = scale

        self.conv_in = nn.Conv2d(in_chans, num_feat, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(num_feat) for _ in range(num_blocks)]
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.drop_paths = nn.ModuleList([DropPath(dpr[i]) for i in range(num_blocks)])
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == "pixelshuffle":
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale),
            )
        else:
            raise NotImplementedError(f"Upsampler '{upsampler}' not implemented.")

        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shallow = self.conv_in(x)
        out = x_shallow
        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
            out = self.drop_paths[i](out)
        out = self.conv_fuse(out) + x_shallow
        out = self.upsampler(out)
        out = self.conv_out(out)
        return out
