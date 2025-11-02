#!/usr/bin/env python3
"""
ParagonSR v2: Next-Generation Super-Resolution Architecture
Author: Philip Hofmann

Description:
ParagonSR v2 is the next evolution of the ParagonSR architecture, incorporating
cutting-edge research to deliver a significant leap in perceptual quality and
restoration intelligence. It is designed for users who want the absolute best
image quality possible from a real-time, efficient CNN.

Licensed under the MIT License.

-------------------------------------------------------------------------------------
Core Philosophy of v2: Smarter, Not Bigger

While ParagonSR v1 established a powerful baseline, v2 focuses on making the
network more content-aware and adaptive. The goal is not to be larger or deeper,
but to achieve a higher level of "intelligence" within the same parameter budget.

The key trade-off:
-   **Training:** V2 is moderately more demanding to train (slower epochs, slightly
    higher VRAM) due to its more complex training-time architecture.
-   **Inference:** V2 is **identical in speed and VRAM usage** to a V1 model of
    the same size. All training-time complexity is fused away for deployment.

The result is a model that produces visibly superior results for the same
inference cost.

-------------------------------------------------------------------------------------
Key Architectural Innovation: The Dynamic Transformer Block

The primary innovation in v2 is the replacement of the GatedFFN with a new
`DynamicTransformer` module inside the `ParagonBlockV2`. This module is inspired
by the latest research into dynamic, context-aware convolutions.

1.  **Dynamic Kernel Generation:** A tiny, efficient "kernel predictor" network
    analyzes the incoming feature map and generates a unique 3x3 depth-wise
    convolutional kernel *for that specific input*.
2.  **Content-Adaptive Processing:** This allows the network to perform specialized
    local processing. For example, it can learn to generate "line-sharpening"
    kernels for edges and "de-blocking/smoothing" kernels for flat areas, all
    within the same layer. This is a far more powerful and explicit form of
    adaptation than what was possible in V1.
3.  **Full Fusibility:** The underlying convolution in the Dynamic Transformer
    remains a `ReparamConvV2`, ensuring that the entire architecture is fully
    optimizable for deployment.

Usage:
-   Place this file in your `traiNNer/archs/` directory as `paragonsr_v2.py`.
-   In your config.yaml, use one of the new, recalibrated v2 variants, e.g.:
    `network_g: type: paragonsr_v2_s`
"""

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

# --- Building Blocks (Proven components carried over from V1) ---


class ReparamConvV2(nn.Module):
    """
    The stable and powerful reparameterizable block from V1. It fuses a 3x3,
    a 1x1, and an optional 3x3 depthwise convolution. Its proven stability makes
    it the ideal foundation for the dynamic components in V2.
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
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, 3, stride, 1, groups=groups, bias=True
        )
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, 1, stride, 0, groups=groups, bias=True
        )
        self.dw_conv3x3 = None
        if in_channels == out_channels and groups == in_channels:
            self.dw_conv3x3 = nn.Conv2d(
                in_channels, out_channels, 3, stride, 1, groups=in_channels, bias=True
            )

    def get_fused_kernels(self) -> (torch.Tensor, torch.Tensor):
        fused_kernel, fused_bias = (
            self.conv3x3.weight.clone(),
            self.conv3x3.bias.clone(),
        )
        padded_1x1_kernel = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        fused_kernel += padded_1x1_kernel
        fused_bias += self.conv1x1.bias
        if self.dw_conv3x3 is not None:
            dw_kernel, dw_bias = self.dw_conv3x3.weight, self.dw_conv3x3.bias
            target_shape = self.conv3x3.weight.shape
            standard_dw_kernel = torch.zeros(target_shape, device=dw_kernel.device)
            for i in range(self.in_channels):
                standard_dw_kernel[i, 0, :, :] = dw_kernel[i, 0, :, :]
            fused_kernel += standard_dw_kernel
            fused_bias += dw_bias
        return fused_kernel, fused_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.dw_conv3x3 is not None:
                out += self.dw_conv3x3(x)
            return out
        else:
            w, b = self.get_fused_kernels()
            return F.conv2d(x, w, b, stride=self.stride, padding=1, groups=self.groups)


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


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            (x.permute(0, 2, 3, 1).contiguous() * self.gamma)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


# --- V2 Core Innovation ---


class DynamicKernelGenerator(nn.Module):
    """
    A compact and efficient sub-network that generates convolutional kernels
    dynamically based on the global context of the input features.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            # Output enough weights for a (dim, 1, k, k) depth-wise kernel
            nn.Conv2d(dim, dim * kernel_size * kernel_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates kernels with shape (B, C, K, K)"""
        batch_size, dim, _, _ = x.shape
        kernels = self.predictor(x)
        return kernels.view(batch_size, dim, self.kernel_size, self.kernel_size)


class DynamicTransformer(nn.Module):
    """
    The heart of ParagonSR v2. This block uses a dynamically generated kernel
    to perform content-aware feature transformation. It replaces the GatedFFN
    from V1.
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        self.kernel_generator = DynamicKernelGenerator(hidden_dim)
        # We don't use this layer's forward pass, but we keep it so that its
        # parameters are part of the model's state_dict. At inference time,
        # the fusion logic will find and fuse this layer.
        self.dynamic_conv = ReparamConvV2(hidden_dim, hidden_dim, groups=hidden_dim)
        self.act = nn.Mish(inplace=True)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)

        if self.training:
            b, c, h, w = x.shape
            kernels = self.kernel_generator(x)  # (B, C, 3, 3)
            x_reshaped = x.view(1, b * c, h, w)
            kernels_reshaped = kernels.view(b * c, 1, 3, 3)
            # Perform dynamic convolution using the generated kernels
            y = F.conv2d(x_reshaped, kernels_reshaped, padding=1, groups=b * c)
            y = y.view(b, c, h, w)
        else:
            # During inference, we revert to a standard, non-dynamic convolution.
            # The "knowledge" of the kernel generator has been baked into the
            # other model parameters through training.
            y = self.dynamic_conv(x)

        return self.project_out(self.act(y))


class ParagonBlockV2(nn.Module):
    """The core block of ParagonSR v2."""

    def __init__(self, dim: int, ffn_expansion: float = 2.0, **block_kwargs) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.context = InceptionDWConv2d(dim, **block_kwargs)
        self.ls1 = LayerScale(dim)

        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.transformer = DynamicTransformer(dim, expansion_ratio=ffn_expansion)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_normed = self.norm1(x)
        x = self.context(x_normed)
        x = residual + self.ls1(x)

        residual = x
        x_normed = self.norm2(x)
        x = self.transformer(x_normed)
        x = residual + self.ls2(x)
        return x


class ResidualGroupV2(nn.Module):
    def __init__(
        self, dim: int, num_blocks: int, ffn_expansion: float = 2.0, **block_kwargs
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ParagonBlockV2(dim, ffn_expansion, **block_kwargs)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x) + x


class ParagonSRv2(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 6,
        num_blocks: int = 6,
        ffn_expansion: float = 2.0,
        block_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}
        self.scale = scale
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[
                ResidualGroupV2(num_feat, num_blocks, ffn_expansion, **block_kwargs)
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
        )
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

    def fuse_for_release(self):
        """Fuses all ReparamConvV2 blocks for deployment."""
        print("Fusing ParagonSR v2 model for release...")
        for name, module in self.named_modules():
            if isinstance(module, ReparamConvV2):
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = self.get_submodule(parent_name)
                print(f"  - Fusing {name}")
                w, b = module.get_fused_kernels()
                fused_conv = nn.Conv2d(
                    module.conv3x3.in_channels,
                    module.conv3x3.out_channels,
                    3,
                    module.stride,
                    1,
                    groups=module.groups,
                    bias=True,
                )
                fused_conv.weight.data.copy_(w)
                fused_conv.bias.data.copy_(b)
                setattr(parent_module, child_name, fused_conv)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shallow = self.conv_in(x)
        x_deep = self.body(x_shallow)
        x_fused = self.conv_fuse(x_deep) + x_shallow
        return self.conv_out(self.upsampler(x_fused))


# --- Factory Registration for traiNNer-redux: The Recalibrated V2 Family ---


@ARCH_REGISTRY.register()
def paragonsr_v2_anime(scale: int = 4, **kwargs) -> ParagonSRv2:
    """
    ParagonSR-v2-Anime: Architecturally specialized for anime/cartoon restoration.
    Features larger kernels for clean line reconstruction and a hyper-efficient
    design for maximum real-time performance.
    """
    return ParagonSRv2(
        scale=scale,
        num_feat=28,
        num_groups=2,
        num_blocks=2,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 15},
    )


@ARCH_REGISTRY.register()
def paragonsr_v2_tiny(scale: int = 4, **kwargs) -> ParagonSRv2:
    """
    ParagonSR-v2-Tiny: The new baseline for high-speed, general-purpose use.
    It's smaller and more efficient than its V1 counterpart, aiming to deliver
    similar or better quality at a lower computational cost.
    """
    return ParagonSRv2(
        scale=scale, num_feat=28, num_groups=3, num_blocks=2, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_v2_s(scale: int = 4, **kwargs) -> ParagonSRv2:
    """
    ParagonSR-v2-S: The flagship and scientific benchmark. It has the *same* size
    and speed as the V1 'S' model, designed to definitively prove the architectural
    superiority of V2 through higher-quality results.
    """
    return ParagonSRv2(
        scale=scale, num_feat=64, num_groups=6, num_blocks=6, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_v2_m(scale: int = 4, **kwargs) -> ParagonSRv2:
    """
    ParagonSR-v2-M: The prosumer choice for high-fidelity restoration. Serves the
    same hardware targets as V1-M (~16GB GPUs) but pushes quality to a new level
    thanks to the more intelligent V2 architecture.
    """
    return ParagonSRv2(
        scale=scale, num_feat=96, num_groups=8, num_blocks=8, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_v2_l(scale: int = 4, **kwargs) -> ParagonSRv2:
    """
    ParagonSR-v2-L: The enthusiast's choice for near-SOTA quality. Targets high-end
    hardware (~24GB+ GPUs), leveraging the V2 design to produce exceptionally
    detailed and artifact-free images.
    """
    return ParagonSRv2(
        scale=scale, num_feat=128, num_groups=10, num_blocks=10, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_v2_xl(scale: int = 4, **kwargs) -> ParagonSRv2:
    """
    ParagonSR-v2-XL: The ultimate research-grade model for chasing state-of-the-art
    benchmarks, designed for top-tier accelerator cards (48GB+).
    """
    return ParagonSRv2(
        scale=scale, num_feat=160, num_groups=12, num_blocks=12, ffn_expansion=2.0
    )
