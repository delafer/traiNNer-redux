#!/usr/bin/env python3
"""
ParagonSR2 - Static Paragon (static kernels, fusion-friendly, training-optimized)

This "Static Paragon" variant removes per-sample dynamic kernels and replaces them
with an efficient, static depthwise alternative plus a cheap channel modulation
option. The goal is to substantially reduce training overhead while retaining
the deployment-focused engineering (fusable reparam blocks, channels-last support,
Magic upsampler, and optional GroupNorm for robustness).

Save as: paragonsr2_static_arch.py
Usage in config: network_g: type: paragonsr2_static_s (or _nano / _xs / _m / _l / _xl)

Author: Philip Hofmann (adapted / refactor)
License: MIT (follow original project license)
"""

import warnings
from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

from .resampler import MagicKernelSharp2021Upsample

# -----------------------------------------------------------
# Low-level building blocks
# -----------------------------------------------------------


class ReparamConvV2(nn.Module):
    """
    Reparameterizable convolution block:
      - training-time: parallel 3x3 and 1x1 (and optional depthwise 3x3)
      - eval-time: fused single 3x3 convolution (via get_fused_kernels)

    Keeps bias=True to allow correct fusion.
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

        # optional depthwise branch (only when in_channels == out_channels and groups == in_channels)
        self.dw_conv3x3: nn.Conv2d | None = None
        if in_channels == out_channels and groups == in_channels:
            self.dw_conv3x3 = nn.Conv2d(
                in_channels, out_channels, 3, stride, 1, groups=in_channels, bias=True
            )

    def get_fused_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (weight, bias) for a fused 3x3 conv equivalent to the multi-branch training structure."""
        fused_kernel = self.conv3x3.weight.detach().clone()
        bias3x3 = self.conv3x3.bias
        if bias3x3 is None:
            raise RuntimeError(
                "ReparamConvV2.conv3x3 must have bias=True for correct fusion."
            )
        fused_bias = bias3x3.detach().clone()

        # add padded 1x1 kernel
        padded_1x1_kernel = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        fused_kernel += padded_1x1_kernel
        bias1x1 = self.conv1x1.bias
        if bias1x1 is None:
            raise RuntimeError(
                "ReparamConvV2.conv1x1 must have bias=True for correct fusion."
            )
        fused_bias += bias1x1.detach()

        # add depthwise converted to standard conv format
        if self.dw_conv3x3 is not None:
            dw_kernel = self.dw_conv3x3.weight
            dw_bias = self.dw_conv3x3.bias
            target_shape = self.conv3x3.weight.shape
            standard_dw_kernel = torch.zeros(target_shape, device=dw_kernel.device)
            for i in range(self.in_channels):
                standard_dw_kernel[i, 0, :, :] = dw_kernel[i, 0, :, :]
            fused_kernel += standard_dw_kernel
            if dw_bias is not None:
                fused_bias += dw_bias.detach()

        return fused_kernel, fused_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.dw_conv3x3 is not None:
                out = out + self.dw_conv3x3(x)
            return out
        else:
            w, b = self.get_fused_kernels()
            return F.conv2d(x, w, b, stride=self.stride, padding=1, groups=self.groups)


class ResidualBlock(nn.Module):
    """
    Simple residual block used in HR head:
      conv -> act -> conv + skip
    Optionally includes GroupNorm for numerical robustness (controlled by flag).
    """

    def __init__(self, dim: int, use_norm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            # GroupNorm with group=1 acts like layer normalization per-channel with no affine.
            self.norm1 = nn.GroupNorm(1, dim, affine=False)
            self.norm2 = nn.GroupNorm(1, dim, affine=False)
        else:
            self.norm1 = None
            self.norm2 = None

        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm and (self.norm1 is not None):
            residual = x
            x = self.norm1(x)
            x = self.conv1(x)
            x = self.act(x)
            x = self.norm2(x)
            x = self.conv2(x)
            return residual + x
        else:
            return x + self.conv2(self.act(self.conv1(x)))


class InceptionDWConv2d(nn.Module):
    """
    Efficient multi-scale depthwise branch:
      - split channels into identity + three depthwise conv branches (square, horizontal, vertical)
    Very parameter-efficient and captures anisotropic context (good for SISR textures).
    """

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        gc = max(1, int(in_channels * branch_ratio))
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, (1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, (band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc
        )
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )


class LayerScale(nn.Module):
    """
    Channel-wise learnable scaling. Implemented as a broadcastable 1xC x1x1 parameter
    to avoid permutes and be friendly for channels_last memory format.
    """

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        # shape (1, C, 1, 1)
        self.gamma = nn.Parameter(
            torch.full((1, dim, 1, 1), float(init_values), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# -----------------------------------------------------------
# Static adaptive modules (cheap adaptivity only)
# -----------------------------------------------------------


class CheapChannelModulation(nn.Module):
    """
    Very cheap channel-wise modulation (SE-like) that provides some adaptivity
    without heavy per-sample kernels. Good speed/benefit tradeoff.
    """

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        inner = max(1, dim // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class StaticDepthwiseTransformer(nn.Module):
    """
    Static 'transformer' replacement for the dynamic transformer:
    - small mixer implemented with grouped / depthwise convs + channel modulation.
    - Fully static weights: cheap to compute and fusable.
    - Designed to be an affordable alternative to per-sample dynamic kernels.
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 2.0,
        use_channel_mod: bool = True,
        group_depthwise: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.project_in = nn.Conv2d(dim, hidden_dim, 1, bias=True)

        # depthwise grouped processing for spatial mixing
        # group_depthwise controls group granularity (1 = full depthwise per channel)
        groups = max(1, group_depthwise)
        self.dw_mixer = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=True
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True),
        )

        self.use_channel_mod = use_channel_mod
        self.channel_mod = (
            CheapChannelModulation(hidden_dim) if use_channel_mod else nn.Identity()
        )

        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dw_mixer(x)
        if self.use_channel_mod:
            x = self.channel_mod(x)
        return self.project_out(x)


# -----------------------------------------------------------
# Mid-level blocks: ParagonBlock / ResidualGroup
# -----------------------------------------------------------


class ParagonBlockStatic(nn.Module):
    """
    Core block for the static Paragon:
      - Inception-style depthwise context (cheap spatial features)
      - LayerScale stabilization
      - StaticDepthwiseTransformer for lightweight global mixing
      - Residual connections around both paths for stability
    """

    def __init__(
        self,
        dim: int,
        ffn_expansion: float = 2.0,
        use_norm: bool = False,
        use_channel_mod: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.context = InceptionDWConv2d(dim, **kwargs)
        self.ls1 = LayerScale(dim)
        self.transformer = StaticDepthwiseTransformer(
            dim, expansion_ratio=ffn_expansion, use_channel_mod=use_channel_mod
        )
        self.ls2 = LayerScale(dim)
        self.use_norm = use_norm

        if use_norm:
            # Optional lightweight GroupNorm at block output for numeric stability (no affine)
            self.gn = nn.GroupNorm(1, dim, affine=False)
        else:
            self.gn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.context(x)
        x = residual + self.ls1(x)

        residual = x
        x = self.transformer(x)
        x = residual + self.ls2(x)

        if self.gn is not None:
            x = self.gn(x)
        return x


class ResidualGroupStatic(nn.Module):
    """
    Stack of ParagonBlockStatic with a residual around the group (LR space).
    """

    def __init__(
        self, dim: int, num_blocks: int, ffn_expansion: float = 2.0, **kwargs
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ParagonBlockStatic(dim, ffn_expansion, **kwargs)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x) + x


# -----------------------------------------------------------
# Main Static Paragon network
# -----------------------------------------------------------


class ParagonSR2Static(nn.Module):
    """
    Static ParagonSR2 implementation.

    Differences to dynamic original:
    - No per-sample dynamic kernels.
    - Uses StaticDepthwiseTransformer + CheapChannelModulation as a faster alternative.
    - Maintains ReparamConvV2 and MagicKernelSharp2021Upsample for inference fidelity.
    - Optional GroupNorm (use_norm) for robustness if desired.
    """

    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 6,
        num_blocks: int = 6,
        ffn_expansion: float = 2.0,
        block_kwargs: dict | None = None,
        upsampler_alpha: float = 0.5,
        hr_blocks: int = 1,
        # performance flags:
        use_channels_last: bool = True,
        fast_body_mode: bool = False,
        use_norm: bool = False,
        use_channel_mod: bool = True,
    ) -> None:
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}

        self.scale = scale
        self.upsampler_alpha = float(max(0.0, min(float(upsampler_alpha), 1.0)))
        self.hr_blocks = max(int(hr_blocks), 0)

        # Fast body mode reduces groups/blocks to speed training (useful during phase 1)
        if fast_body_mode:
            num_groups = max(1, num_groups // 2)
            num_blocks = max(1, num_blocks // 2)

        # Shallow feature extraction
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1, bias=True)

        # Body (LR space)
        self.body = nn.Sequential(
            *[
                ResidualGroupStatic(
                    num_feat,
                    num_blocks,
                    ffn_expansion,
                    use_norm=use_norm,
                    use_channel_mod=use_channel_mod,
                    **block_kwargs,
                )
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # Magic upsampler (keeps your MagicKernelSharp2021 implementation)
        self.magic_upsampler = MagicKernelSharp2021Upsample(
            in_channels=num_feat, alpha=self.upsampler_alpha
        )

        # HR refinement
        self.hr_conv_in = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        if self.hr_blocks > 0:
            self.hr_head = nn.Sequential(
                *[
                    ResidualBlock(num_feat, use_norm=use_norm)
                    for _ in range(self.hr_blocks)
                ]
            )
        else:
            self.hr_head = nn.Identity()

        # Output
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, 1, 1, bias=True)

        # memory format optimization
        self.use_channels_last = use_channels_last and torch.cuda.is_available()
        if self.use_channels_last:
            # convert parameter tensors to channels_last contiguous for better memory locality
            try:
                for module in self.modules():
                    if hasattr(module, "weight") and module.weight is not None:
                        # only float tensors make sense here
                        if module.weight.dtype in (torch.float32, torch.float16):
                            module.weight.data = module.weight.contiguous(
                                memory_format=torch.channels_last
                            )
            except Exception:
                # be defensive — some modules may not support channels_last layout changes
                pass

    # Fusion helper for deployment
    def fuse_for_release(self) -> "ParagonSR2Static":
        """
        Fuse any ReparamConvV2 blocks to plain Conv2d for faster inference exports.
        Also prepares the model to run deterministically by ensuring no dynamic parts remain.
        """
        print("Fusing ParagonSR2Static for release...")
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
                with torch.no_grad():
                    fused_conv.weight.copy_(w)
                    if fused_conv.bias is None:
                        raise RuntimeError("Expected fused_conv to have bias True")
                    fused_conv.bias.copy_(b)
                setattr(parent_module, child_name, fused_conv)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channels_last friendly path for fp16 training if requested
        if self.use_channels_last and x.dtype == torch.float16:
            x = x.contiguous(memory_format=torch.channels_last)

        # core forward
        x_shallow = self.conv_in(x)
        x_deep = self.body(x_shallow)
        x_fused = self.conv_fuse(x_deep) + x_shallow

        x_upsampled = self.magic_upsampler(x_fused, scale_factor=self.scale)

        h = self.hr_conv_in(x_upsampled)
        h = self.hr_head(h)

        return self.conv_out(h)


# -----------------------------------------------------------
# Factory variants (registered)
# -----------------------------------------------------------


@ARCH_REGISTRY.register()
def paragonsr2_static_micro(scale: int = 4, **kwargs) -> ParagonSR2Static:
    """
    Micro variant: extremely small / fast. Useful for fast experimentation
    or real-time upscaling where extreme latency constraints exist.
    Designed to still learn compression artifacts if those appear in LR data.
    """
    return ParagonSR2Static(
        scale=scale,
        num_feat=16,
        num_groups=1,
        num_blocks=1,
        ffn_expansion=1.2,
        block_kwargs={"band_kernel_size": 7},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 0),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", True),
        use_norm=kwargs.get("use_norm", False),
        use_channel_mod=kwargs.get("use_channel_mod", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_static_nano(scale: int = 4, **kwargs) -> ParagonSR2Static:
    """
    Nano variant: extremely small, best for quick prototyping and very limited VRAM.
    Uses fast_body_mode=True by default for fastest training.
    """
    return ParagonSR2Static(
        scale=scale,
        num_feat=24,
        num_groups=2,
        num_blocks=2,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 9},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 1),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", True),
        use_norm=kwargs.get("use_norm", True),
        use_channel_mod=kwargs.get("use_channel_mod", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_static_xs(scale: int = 4, **kwargs) -> ParagonSR2Static:
    """XS (extra-small) variant: slightly larger than nano, still training-friendly."""
    return ParagonSR2Static(
        scale=scale,
        num_feat=32,
        num_groups=2,
        num_blocks=3,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 11},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 1),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", True),
        use_norm=kwargs.get("use_norm", True),
        use_channel_mod=kwargs.get("use_channel_mod", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_static_s(scale: int = 4, **kwargs) -> ParagonSR2Static:
    """S: training-friendly variant recommended for RTX 3060 class hardware."""
    return ParagonSR2Static(
        scale=scale,
        num_feat=48,
        num_groups=3,
        num_blocks=4,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 11},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 2),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get(
            "fast_body_mode", True
        ),  # enable for faster training by default
        use_norm=kwargs.get(
            "use_norm", True
        ),  # numeric stability often helps for GAN/perceptual training
        use_channel_mod=kwargs.get("use_channel_mod", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_static_m(scale: int = 4, **kwargs) -> ParagonSR2Static:
    """M: medium quality variant. Slightly heavier, recommended with full training mode."""
    return ParagonSR2Static(
        scale=scale,
        num_feat=64,
        num_groups=4,
        num_blocks=6,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 13},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 2),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", False),
        use_norm=kwargs.get("use_norm", True),
        use_channel_mod=kwargs.get("use_channel_mod", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_static_l(scale: int = 4, **kwargs) -> ParagonSR2Static:
    """L: large variant for users with more VRAM; prefer fast_body_mode=False."""
    return ParagonSR2Static(
        scale=scale,
        num_feat=96,
        num_groups=6,
        num_blocks=8,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 15},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 3),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", False),
        use_norm=kwargs.get("use_norm", True),
        use_channel_mod=kwargs.get("use_channel_mod", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_static_xl(scale: int = 4, **kwargs) -> ParagonSR2Static:
    """XL: research-grade variant. Only use if you have >=24GB VRAM or distributed training."""
    return ParagonSR2Static(
        scale=scale,
        num_feat=128,
        num_groups=8,
        num_blocks=10,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 15},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 3),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", False),
        use_norm=kwargs.get("use_norm", True),
        use_channel_mod=kwargs.get("use_channel_mod", True),
    )
