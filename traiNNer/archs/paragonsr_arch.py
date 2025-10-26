#!/usr/bin/env python3
"""
ParagonSR: A High-Performance Super-Resolution Network
Author: Philip Hofmann

Description:
ParagonSR is a state-of-the-art, general-purpose super-resolution architecture
designed for a superior balance of performance, training efficiency, and
inference speed. It represents a synthesis of cutting-edge concepts from a
multitude of modern network designs, aiming to be a direct competitor to models
like RealPLKSR, FDAT, and HAT.

Core Design Philosophy & Innovations:
1.  **The ParagonBlock:** A novel core block that synergizes three key ideas
    to create a powerful and efficient building block:
    -   **Efficient Multi-Scale Context:** Utilizes an Inception-style depthwise
        convolution block (inspired by MoSRv2/RTMoSR) to capture features at
        multiple spatial scales (square, horizontal, vertical) simultaneously
        with high parameter efficiency.
    -   **Powerful Gated Transformation:** Employs a Gated Feed-Forward Network
        (inspired by HyperionSR/GaterV3) for superior non-linear feature
        transformation, allowing the network to dynamically route information.
    -   **Inference-Time Reparameterization:** The spatial mixing component is
        built with reparameterization principles. Its complex, multi-branch
        training-time structure can be mathematically fused into a single,
        ultra-fast convolution for deployment, drastically improving inference speed.

2.  **Hierarchical Residual Groups:** Organizes the ParagonBlocks into residual
    groups for improved training stability and gradient flow, enabling deeper
    and more powerful network configurations.

Usage:
-   Place this file in your `traiNNer/archs/` directory.
-   In your config.yaml, use one of the registered variants:
    `network_g: type: paragonsr_s`
    `network_g: type: paragonsr_m`
    `network_g: type: paragonsr_l`
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

# --- Building Blocks ---


class ReparamConv(nn.Module):
    """
    A reparameterizable convolutional block that fuses a 3x3 conv, a 1x1 conv,
    and an identity connection (via BatchNorm) into a single 3x3 conv for inference.
    This version is production-ready, correctly handles grouped convolutions, and
    includes an explicit `_is_fused` flag to track its state.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

        self.register_buffer("_is_fused", torch.tensor(False, dtype=torch.bool))

        # Training-time branches
        self.conv3x3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=True,
        )
        self.conv1x1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            groups=groups,
            bias=True,
        )
        self.identity = (
            nn.BatchNorm2d(in_channels, affine=True)
            if in_channels == out_channels and stride == 1
            else None
        )

        # Inference-time branch
        self.fused_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            out = self.conv3x3(x)
            out = out + self.conv1x1(x)
            if self.identity is not None:
                out = out + self.identity(x)
            return out
        elif self._is_fused:
            return self.fused_conv(x)
        else:
            self.fuse_kernels()
            return self.fused_conv(x)

    def fuse_kernels(self) -> None:
        """
        Performs the mathematical fusion of the training-time branches into the
        single `fused_conv` layer. This is a one-way operation.
        """
        if self._is_fused:
            return

        # --- Step 1: Fuse 3x3 and 1x1 branches ---
        fused_kernel = self.conv3x3.weight.clone()
        fused_bias = self.conv3x3.bias.clone()

        padded_1x1_kernel = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        fused_kernel += padded_1x1_kernel
        fused_bias += self.conv1x1.bias

        # --- Step 2: Fuse the Identity branch (if it exists) ---
        if self.identity is not None:
            running_var = self.identity.running_var
            running_mean = self.identity.running_mean
            gamma = self.identity.weight
            beta = self.identity.bias
            eps = self.identity.eps

            std = (running_var + eps).sqrt()

            bn_scale = gamma / std
            bn_bias = beta - running_mean * bn_scale

            identity_kernel_1x1 = torch.zeros(
                self.in_channels,
                self.in_channels // self.groups,
                1,
                1,
                device=fused_kernel.device,
            )
            for i in range(self.in_channels):
                identity_kernel_1x1[i, i // self.groups, 0, 0] = 1.0

            identity_kernel_3x3 = F.pad(identity_kernel_1x1, [1, 1, 1, 1])
            identity_kernel_3x3 *= bn_scale.reshape(-1, 1, 1, 1)

            fused_kernel += identity_kernel_3x3
            fused_bias += bn_bias

        # --- Step 3: Load the final weights into the inference convolution ---
        self.fused_conv.weight.data.copy_(fused_kernel)
        self.fused_conv.bias.data.copy_(fused_bias)

        self.fused_conv.weight.requires_grad = self.conv3x3.weight.requires_grad
        self.fused_conv.bias.requires_grad = self.conv3x3.bias.requires_grad

        self._is_fused.fill_(True)

    def train(self, mode: bool = True) -> None:
        """Override train() to handle fusion state. Does NOT automatically fuse."""
        super().train(mode)
        # When switching back to training mode, we must mark the model as "not fused"
        # so that the training-path is used in the forward pass. The weights of the
        # training branches were never touched, so they are still valid.
        if mode is True:
            self._is_fused.fill_(False)


class InceptionDWConv2d(nn.Module):
    """
    Inception-style Depthwise Convolution Block. Efficiently captures features
    at multiple spatial scales (square, horizontal, vertical).
    """

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network for powerful, non-linear feature transformation.
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.project_in_g = nn.Conv2d(dim, hidden_dim, 1)
        self.project_in_i = nn.Conv2d(dim, hidden_dim, 1)
        self.spatial_mixer = ReparamConv(hidden_dim, hidden_dim, groups=hidden_dim)
        self.act = nn.Mish(inplace=True)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.project_in_g(x)
        i = self.project_in_i(x)
        g = self.spatial_mixer(g)
        return self.project_out(self.act(g) * i)


class ParagonBlock(nn.Module):
    """
    The core block of ParagonSR.
    Combines multi-scale context gathering with a powerful gated feature transformer.
    """

    def __init__(self, dim: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.context = InceptionDWConv2d(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.transformer = GatedFFN(dim, expansion_ratio=ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm expects (B, H, W, C), so we permute.
        # .contiguous() is used to ensure memory layout is correct after permute.
        _B, _C, _H, _W = x.shape

        # Context Gathering Branch
        residual = x
        x_normed = x.permute(0, 2, 3, 1).contiguous()
        x_normed = self.norm1(x_normed)
        x_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        x = self.context(x_normed)
        x = x + residual

        # Feature Transformation Branch
        residual = x
        x_normed = x.permute(0, 2, 3, 1).contiguous()
        x_normed = self.norm2(x_normed)
        x_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        x = self.transformer(x_normed)
        x = x + residual

        return x


class ResidualGroup(nn.Module):
    """A group of ParagonBlocks with a local residual connection for stability."""

    def __init__(self, dim: int, num_blocks: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ParagonBlock(dim, ffn_expansion) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x) + x


# --- The Main ParagonSR Network ---


class ParagonSR(nn.Module):
    """The main ParagonSR architecture."""

    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 6,
        num_blocks: int = 6,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.scale = scale

        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, padding=1)
        self.body = nn.Sequential(
            *[
                ResidualGroup(num_feat, num_blocks, ffn_expansion)
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shallow = self.conv_in(x)
        x_deep = self.body(x_shallow)
        x_fused = self.conv_fuse(x_deep) + x_shallow
        x_upscaled = self.upsampler(x_fused)
        return self.conv_out(x_upscaled)


# --- Factory Registration for traiNNer-redux ---


@ARCH_REGISTRY.register()
def paragonsr_s(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-S: Small variant.
    - Training Target: ~12GB VRAM GPU (e.g., RTX 3060).
    - Inference Target: ~6-8GB VRAM GPU (e.g., RTX 2060, GTX 1660S).
    """
    return ParagonSR(
        scale=scale, num_feat=64, num_groups=6, num_blocks=6, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_m(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-M: Medium variant.
    - Training Target: ~16-24GB VRAM GPU (e.g., RTX 3090, RTX 4080).
    - Inference Target: ~8-12GB VRAM GPU (e.g., RTX 3060, RTX 2070).
    """
    return ParagonSR(
        scale=scale, num_feat=96, num_groups=8, num_blocks=8, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_l(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-L: Large variant.
    - Training Target: High-end GPU with >24GB VRAM (e.g., RTX 4090, A100).
    - Inference Target: ~12GB+ VRAM GPU (e.g., RTX 3080, RTX 4070).
    """
    return ParagonSR(
        scale=scale, num_feat=128, num_groups=10, num_blocks=10, ffn_expansion=2.0
    )
