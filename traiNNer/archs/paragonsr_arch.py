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
    `network_g: type: paragonsr_tiny`
    `network_g: type: paragonsr_xs`
    `network_g: type: paragonsr_s`
    `network_g: type: paragonsr_m`
    `network_g: type: paragonsr_l`
    `network_g: type: paragonsr_xl`
"""

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

# --- Building Blocks ---


class ReparamConv(nn.Module):
    """
    A stable, reparameterizable convolutional block that fuses a 3x3 and a 1x1
    branch into a single 3x3 conv for inference.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.stride, self.groups = (
            in_channels,
            out_channels,
            stride,
            groups,
        )
        self.register_buffer("_is_fused", torch.tensor(False, dtype=torch.bool))
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, 3, stride, 1, groups=groups, bias=True
        )
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, 1, stride, 0, groups=groups, bias=True
        )
        self.fused_conv = nn.Conv2d(
            in_channels, out_channels, 3, stride, 1, groups=groups, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.conv3x3(x) + self.conv1x1(x)
        elif self._is_fused:
            return self.fused_conv(x)
        else:
            self.fuse_kernels()
            return self.fused_conv(x)

    def fuse_kernels(self) -> None:
        if self._is_fused:
            return
        fused_kernel, fused_bias = (
            self.conv3x3.weight.clone(),
            self.conv3x3.bias.clone(),
        )
        padded_1x1_kernel = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        fused_kernel += padded_1x1_kernel
        fused_bias += self.conv1x1.bias
        self.fused_conv.weight.data.copy_(fused_kernel)
        self.fused_conv.bias.data.copy_(fused_bias)
        self._is_fused.fill_(True)

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self._is_fused:
            self._is_fused.fill_(False)
        elif not mode and not self._is_fused:
            self.fuse_kernels()


class InceptionDWConv2d(nn.Module):
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
            gc, gc, (1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, (band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )


class GatedFFN(nn.Module):
    def __init__(self, dim: int, expansion_ratio: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.project_in_g = nn.Conv2d(dim, hidden_dim, 1)
        self.project_in_i = nn.Conv2d(dim, hidden_dim, 1)
        self.spatial_mixer = ReparamConv(hidden_dim, hidden_dim, groups=hidden_dim)
        self.act = nn.Mish(inplace=True)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, i = self.project_in_g(x), self.project_in_i(x)
        g = self.spatial_mixer(g)
        return self.project_out(self.act(g) * i)


class ParagonBlock(nn.Module):
    def __init__(self, dim: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.context = InceptionDWConv2d(dim)
        self.transformer = GatedFFN(dim, expansion_ratio=ffn_expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _B, _C, _H, _W = x.shape
        residual = x
        x_normed = (
            self.norm1(x.permute(0, 2, 3, 1).contiguous())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x = self.context(x_normed) + residual
        residual = x
        x_normed = (
            self.norm2(x.permute(0, 2, 3, 1).contiguous())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x = self.transformer(x_normed) + residual
        return x


class ResidualGroup(nn.Module):
    def __init__(self, dim: int, num_blocks: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ParagonBlock(dim, ffn_expansion) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x) + x


class ParagonSR(nn.Module):
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
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[
                ResidualGroup(num_feat, num_blocks, ffn_expansion)
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
        )
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shallow = self.conv_in(x)
        x_deep = self.body(x_shallow)
        x_fused = self.conv_fuse(x_deep) + x_shallow
        return self.conv_out(self.upsampler(x_fused))


# --- Factory Registration for traiNNer-redux: The Complete Family ---


@ARCH_REGISTRY.register()
def paragonsr_tiny(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-Tiny: Ultra-lightweight variant for high-speed/real-time use.
    - Use Case: Video upscaling, fast previews where quality is secondary to speed.
    - Training Target: Any GPU with ~4-6GB VRAM. Very fast to train.
    - Inference Target: Any modern GPU/CPU; suitable for real-time applications.
    """
    return ParagonSR(
        scale=scale, num_feat=32, num_groups=3, num_blocks=3, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_xs(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-XS: Extra-Small variant. A wide but shallow design.
    - Use Case: General-purpose upscaling on low-end hardware.
    - Training Target: ~6-8GB VRAM GPU (e.g., RTX 2060, GTX 1660S). Fast to train.
    - Inference Target: ~4-6GB VRAM GPU (e.g., GTX 1060).
    """
    return ParagonSR(
        scale=scale, num_feat=48, num_groups=4, num_blocks=4, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_s(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-S: Small variant, the recommended flagship model.
    - Use Case: High-quality general-purpose upscaling.
    - Training Target: ~12GB VRAM GPU (e.g., RTX 3060).
    - Inference Target: Most GPUs with ~6-8GB VRAM (e.g., RTX 2070).
    """
    return ParagonSR(
        scale=scale, num_feat=64, num_groups=6, num_blocks=6, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_m(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-M: Medium variant, for prosumer hardware.
    - Use Case: Higher-quality upscaling for users with strong GPUs.
    - Training Target: ~16-24GB VRAM GPU (e.g., RTX 3090, RTX 4080).
    - Inference Target: GPUs with ~8-12GB VRAM (e.g., RTX 3060).
    """
    return ParagonSR(
        scale=scale, num_feat=96, num_groups=8, num_blocks=8, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_l(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-L: Large variant, for high-end enthusiast hardware.
    - Use Case: Near-SOTA quality for users with top-tier hardware.
    - Training Target: >24GB VRAM GPU (e.g., RTX 4090, RTX 3090 with small batch).
    - Inference Target: High-end GPUs with ~12GB+ VRAM (e.g., RTX 3080, RTX 4070).
    """
    return ParagonSR(
        scale=scale, num_feat=128, num_groups=10, num_blocks=10, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_xl(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-XL: Extra-Large variant for researchers and enthusiasts.
    - Use Case: Pushing the absolute limits of quality, regardless of cost.
    - Training Target: High-VRAM accelerator cards (e.g., 48GB+ A100, H100).
    - Inference Target: Flagship GPUs with >24GB VRAM (e.g., RTX 4090).
    """
    return ParagonSR(
        scale=scale, num_feat=160, num_groups=12, num_blocks=12, ffn_expansion=2.0
    )
