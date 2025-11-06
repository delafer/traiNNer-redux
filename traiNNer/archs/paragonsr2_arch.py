#!/usr/bin/env python3
"""
ParagonSR2: A Refined, Deployment-Ready Super-Resolution Architecture
Author: Philip Hofmann

Description:
ParagonSR2 is the definitive evolution of the ParagonSR series. It is a
synergistic blend of the V2 architecture's content-aware intelligence with a
series of targeted refinements aimed at maximizing training speed, numerical
stability, and deployment robustness for ONNX, TensorRT, and INT8 quantization.

Licensed under the MIT License.

-------------------------------------------------------------------------------------
Core Philosophy of 2: Professional-Grade Practicality

This version prioritizes the trade-offs that matter for real-world application.
It is designed to deliver state-of-the-art perceptual quality while ensuring the
resulting model is fast, efficient, and easy to deploy.

Key Refinements in 2:
1.  **Normalization-Free Blocks:** `GroupNorm` layers have been removed from the
    core blocks. This significantly simplifies the data path, making the model
    faster to train and far more compatible with post-training quantization (INT8).
    Training stability is maintained through the robust residual structure and
    the use of LayerScale.
2.  **Standardized Activation:** The `Mish` activation function has been replaced
    with `LeakyReLU`. This provides a tangible speed boost and guarantees optimal,
    hardware-accelerated support across all major inference engines.
3.  **Superior Upsampling:** The architecture retains the "Magic-Conv" upsampler,
    using the Magic Kernel Sharp 2021 algorithm to ensure the final output is
    free from the common artifacts associated with other methods like PixelShuffle.

The result is a model that is not only powerful in theory but also practical
and robust in application.

Usage:
-   Place this file in your `traiNNer/archs/` directory.
-   Ensure your `resampler.py` file is in the same directory.
-   In your config.yaml, use one of the v2 variants, e.g.:
    `network_g: type: paragonsr_v2_s`
"""

import warnings
from typing import Optional, cast

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

from .resampler import MagicKernelSharp2021Upsample

# --- Building Blocks (Proven components from V1) ---


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

    def get_fused_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs the mathematical fusion of the training-time branches."""
        fused_kernel = self.conv3x3.weight.detach().clone()
        bias3x3 = self.conv3x3.bias
        if bias3x3 is None:
            raise RuntimeError("ReparamConvV2.conv3x3 must use bias=True for fusion.")
        fused_bias = bias3x3.detach().clone()

        padded_1x1_kernel = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        fused_kernel += padded_1x1_kernel
        bias1x1 = self.conv1x1.bias
        if bias1x1 is None:
            raise RuntimeError("ReparamConvV2.conv1x1 must use bias=True for fusion.")
        fused_bias += bias1x1.detach()

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
        """Uses multi-branch for training and stateless on-the-fly fusion for eval."""
        if self.training:
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.dw_conv3x3 is not None:
                out += self.dw_conv3x3(x)
            return out
        else:
            w, b = self.get_fused_kernels()
            return F.conv2d(x, w, b, stride=self.stride, padding=1, groups=self.groups)


class InceptionDWConv2d(nn.Module):
    """
    Efficiently captures features at multiple spatial scales (square, horizontal,
    vertical) with high parameter efficiency, inspired by MoSRv2/RTMoSR.
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
    A key stabilization technique for deep networks. It applies a learnable
    scaling factor to the output of a residual block, preventing signal explosion.
    """

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            (x.permute(0, 2, 3, 1).contiguous() * self.gamma)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


# --- V2.1 Core Innovation ---


class DynamicKernelGenerator(nn.Module):
    """
    A compact sub-network that generates convolutional kernels dynamically
    based on the global context of the input features.
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
        return kernels.reshape(batch_size, dim, self.kernel_size, self.kernel_size)


class DynamicTransformer(nn.Module):
    """
    Content-adaptive transformer block with a deploy-friendly static fallback.

    During training, it uses per-sample kernels generated on-the-fly to maximize
    expressive power. A depthwise kernel tracker aggregates these dynamic kernels,
    enabling a deterministic static inference path (required for ONNX/TensorRT
    export) without sacrificing the benefits of dynamic training.
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        self.kernel_generator: DynamicKernelGenerator | None = DynamicKernelGenerator(
            hidden_dim
        )

        self.kernel_size = 3
        self.padding = self.kernel_size // 2
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1)

        # Deployment controls
        self.dynamic_inference = False
        self.track_running_stats = True
        self.kernel_momentum = 0.05
        self.static_mode = False

        identity = torch.zeros(
            hidden_dim, 1, self.kernel_size, self.kernel_size, dtype=torch.float32
        )
        identity[:, 0, self.padding, self.padding] = 1.0
        self.register_buffer("identity_kernel", identity, persistent=True)
        self.register_buffer(
            "tracked_kernel",
            torch.zeros_like(identity),
            persistent=True,
        )
        self.register_buffer(
            "tracked_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self._warned_identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        use_dynamic = self.training or self.dynamic_inference
        if self.static_mode:
            use_dynamic = False

        if use_dynamic:
            kernel_generator = self.kernel_generator
            if kernel_generator is None:
                raise RuntimeError(
                    "Dynamic inference requested but the kernel generator was removed. "
                    "Call `enable_dynamic_inference(False)` before export or keep the generator intact."
                )
            b, c, _h, _w = x.shape
            kernels = kernel_generator(x)  # (B, C, 3, 3)
            y = self._apply_dynamic_kernel(x, kernels, batch_size=b, channels=c)
            if self.training and self.track_running_stats:
                self._update_tracked_kernel(kernels)
        else:
            y = self._apply_static_kernel(x)

        return self.project_out(self.act(y))

    def _apply_dynamic_kernel(
        self,
        x: torch.Tensor,
        kernels: torch.Tensor,
        batch_size: int,
        channels: int,
    ) -> torch.Tensor:
        x_reshaped = x.reshape(1, batch_size * channels, x.shape[2], x.shape[3])
        kernels_reshaped = kernels.reshape(
            batch_size * channels, 1, self.kernel_size, self.kernel_size
        )
        y = F.conv2d(
            x_reshaped,
            kernels_reshaped,
            padding=self.padding,
            groups=batch_size * channels,
        )
        return y.reshape(batch_size, channels, x.shape[2], x.shape[3])

    def _apply_static_kernel(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._get_tracked_kernel().to(dtype=x.dtype, device=x.device)
        return F.conv2d(
            x,
            kernel,
            padding=self.padding,
            groups=x.shape[1],
        )

    def _get_tracked_kernel(self) -> torch.Tensor:
        tracked_initialized = cast(torch.Tensor, self.tracked_initialized)
        tracked_kernel = cast(torch.Tensor, self.tracked_kernel)
        identity_kernel = cast(torch.Tensor, self.identity_kernel)
        if bool(tracked_initialized.item()):
            self._warned_identity = False
            return tracked_kernel
        if not self.training and not self._warned_identity:
            warnings.warn(
                "DynamicTransformer is falling back to the identity kernel because "
                "no running statistics were collected. Call `model.train()` for a "
                "few iterations or load a checkpoint with tracked kernels before "
                "exporting/deploying.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_identity = True
        return identity_kernel

    def _update_tracked_kernel(self, kernels: torch.Tensor) -> None:
        with torch.no_grad():
            tracked_kernel = cast(torch.Tensor, self.tracked_kernel)
            tracked_initialized = cast(torch.Tensor, self.tracked_initialized)
            mean_kernel = (
                kernels.mean(dim=0).unsqueeze(1).to(dtype=tracked_kernel.dtype)
            )
            if not bool(tracked_initialized.item()):
                tracked_kernel.copy_(mean_kernel)
                tracked_initialized.fill_(True)
            else:
                momentum = float(self.kernel_momentum)
                if momentum <= 0:
                    tracked_kernel.copy_(mean_kernel)
                else:
                    tracked_kernel.lerp_(mean_kernel, weight=momentum)
            self._warned_identity = False

    def enable_dynamic_inference(self, enabled: bool = True) -> None:
        """
        Toggle dynamic kernels during evaluation.
        """
        self.dynamic_inference = enabled

    def enable_static_mode(self, enabled: bool = True) -> None:
        """
        Force the transformer to use the tracked static kernel regardless of the
        current training/eval mode. Useful when preparing a model for release.
        """
        self.static_mode = enabled

    def set_kernel_momentum(self, momentum: float) -> None:
        """
        Adjust the EMA momentum used to aggregate dynamic kernels.
        """
        if momentum < 0 or momentum > 1:
            raise ValueError("kernel_momentum must be in [0, 1].")
        self.kernel_momentum = float(momentum)

    def reset_tracked_kernel(self) -> None:
        """
        Clear the accumulated static kernel statistics.
        """
        tracked_kernel = cast(torch.Tensor, self.tracked_kernel)
        tracked_initialized = cast(torch.Tensor, self.tracked_initialized)
        tracked_kernel.zero_()
        tracked_initialized.fill_(False)
        self._warned_identity = False

    def export_static_depthwise(self) -> nn.Conv2d:
        """
        Create a depthwise 3x3 convolution initialized with the tracked kernel.
        """
        kernel = self._get_tracked_kernel().detach()
        conv = nn.Conv2d(
            kernel.shape[0],
            kernel.shape[0],
            self.kernel_size,
            padding=self.padding,
            groups=kernel.shape[0],
            bias=False,
        )
        with torch.no_grad():
            conv.weight.copy_(kernel)
        return conv


class ParagonBlockV2(nn.Module):
    """
    The core block of ParagonSR v2.1. This version is normalization-free,
    relying on residual connections and LayerScale for stability. This design
    improves training speed and ONNX/INT8 deployment robustness.
    """

    def __init__(self, dim: int, ffn_expansion: float = 2.0, **block_kwargs) -> None:
        super().__init__()
        self.context = InceptionDWConv2d(dim, **block_kwargs)
        self.ls1 = LayerScale(dim)
        self.transformer = DynamicTransformer(dim, expansion_ratio=ffn_expansion)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.context(x)
        x = residual + self.ls1(x)

        residual = x
        x = self.transformer(x)
        x = residual + self.ls2(x)
        return x


class ResidualGroupV2(nn.Module):
    """A group of ParagonBlocks with a residual connection for stable training."""

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


# --- Main Architecture ---


class ParagonSR2(nn.Module):
    """The complete ParagonSR v2.1 architecture."""

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

        # --- Shallow Feature Extraction ---
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1)

        # --- Deep Feature Extraction (The Body) ---
        self.body = nn.Sequential(
            *[
                ResidualGroupV2(num_feat, num_blocks, ffn_expansion, **block_kwargs)
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # --- Upsampling Block: The "Magic-Conv" ---
        # Using the Magic Kernel Sharp 2021 provides a sharper, cleaner, and more
        # foundationally sound input for the final stage of image reconstruction,
        # avoiding common artifacts from other upsampling methods.
        self.magic_upsampler = MagicKernelSharp2021Upsample(in_channels=num_feat)
        self.upsampler = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # --- Final Image Reconstruction ---
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

    def fuse_for_release(self):
        """Fuses all ReparamConvV2 blocks for deployment."""
        print("Fusing ParagonSR v2.1 model for release...")
        for name, module in self.named_modules():
            if isinstance(module, DynamicTransformer):
                tracked_initialized = cast(torch.Tensor, module.tracked_initialized)
                if not bool(tracked_initialized.item()):
                    warnings.warn(
                        f"{name}: tracked kernel statistics were not initialized before fusion. "
                        "The identity kernel will be used. Run a brief calibration pass before fusion for best quality.",
                        stacklevel=2,
                    )
                module.enable_static_mode(True)
                module.kernel_generator = None
                continue

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
                        raise RuntimeError(
                            "Reparam fusion expects fused_conv to include a bias term."
                        )
                    fused_conv.bias.copy_(b)
                setattr(parent_module, child_name, fused_conv)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shallow = self.conv_in(x)
        x_deep = self.body(x_shallow)
        x_fused = self.conv_fuse(x_deep) + x_shallow
        x_upsampled = self.magic_upsampler(x_fused, scale_factor=self.scale)
        return self.conv_out(self.upsampler(x_upsampled))


# --- Factory Registration for traiNNer-redux: The Recalibrated V2 Family ---


@ARCH_REGISTRY.register()
def paragonsr2_nano(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-Nano: Ultra-light configuration for rapid prototyping and low-VRAM hardware.
    - Target Hardware (Train): ~4GB VRAM GPUs (GTX 1650, RTX 3050 Laptop).
    """
    return ParagonSR2(
        scale=scale,
        num_feat=24,
        num_groups=2,
        num_blocks=2,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 9},
    )


@ARCH_REGISTRY.register()
def paragonsr2_anime(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-Anime: Specialized for animation. Features wider context kernels for clean
    line reconstruction and an efficient design for real-time performance.
    - Target Hardware (Train): ~8-12GB VRAM GPUs (RTX 3060).
    """
    return ParagonSR2(
        scale=scale,
        num_feat=28,
        num_groups=2,
        num_blocks=3,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 15},
    )


@ARCH_REGISTRY.register()
def paragonsr2_tiny(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-Tiny: The ideal starting point for quick tests and low-resource training.
    Excellent for validating a training pipeline.
    - Target Hardware (Train): ~6-8GB VRAM GPUs (GTX 1660S, RTX 3050).
    """
    return ParagonSR2(
        scale=scale, num_feat=32, num_groups=3, num_blocks=2, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr2_s(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-S (Recalibrated): The flagship model, designed for high quality on
    mainstream hardware. It leverages the intelligent V2 architecture to achieve
    superior results within a practical training budget.
    - Target Hardware (Train): ~12GB VRAM GPUs (RTX 3060, RTX 2080 Ti).
    """
    return ParagonSR2(
        scale=scale, num_feat=56, num_groups=5, num_blocks=5, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr2_m(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-M: The prosumer choice for future hardware, offering a significant
    jump in expressive power for higher fidelity restoration.
    - Target Hardware (Train): ~16-24GB VRAM GPUs (RTX 3090, RTX 4080).
    """
    return ParagonSR2(
        scale=scale, num_feat=96, num_groups=8, num_blocks=8, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr2_l(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-L: The enthusiast's choice for near-SOTA quality on high-end hardware.
    - Target Hardware (Train): ~24GB+ VRAM GPUs (RTX 4090).
    """
    return ParagonSR2(
        scale=scale, num_feat=128, num_groups=10, num_blocks=10, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr2_xl(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-XL: The ultimate research-grade model for chasing state-of-the-art
    benchmarks, designed for top-tier accelerator cards.
    - Target Hardware (Train): 48GB+ VRAM (NVIDIA A100, H100).
    """
    return ParagonSR2(
        scale=scale, num_feat=160, num_groups=12, num_blocks=12, ffn_expansion=2.0
    )
