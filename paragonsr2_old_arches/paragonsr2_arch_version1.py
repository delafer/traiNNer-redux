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


class ResidualBlock(nn.Module):
    """
    Lightweight residual block for HR refinement.
    Uses Conv + LeakyReLU + Conv + skip, keeping it simple, fusable,
    and export-friendly (ONNX/TensorRT/INT8).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


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
    """The complete ParagonSR v2.1 architecture with HR refinement head."""

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
    ) -> None:
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}
        self.scale = scale

        # Clamp and store upsampler_alpha (0.0 = no sharpen, 1.0 = full MagicSharp behavior).
        upsampler_alpha = float(upsampler_alpha)
        upsampler_alpha = max(upsampler_alpha, 0.0)
        upsampler_alpha = min(upsampler_alpha, 1.0)
        self.upsampler_alpha = upsampler_alpha

        # HR refinement depth (non-negative int). Acts as a local corrector, not a second backbone.
        self.hr_blocks = max(int(hr_blocks), 0)

        # --- Shallow Feature Extraction ---
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1)

        # --- Deep Feature Extraction (The Body, LR space) ---
        self.body = nn.Sequential(
            *[
                ResidualGroupV2(num_feat, num_blocks, ffn_expansion, **block_kwargs)
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # --- Upsampling Block: The "Magic-Conv" ---
        # Using the Magic Kernel Sharp 2021 with moderated alpha provides a sharp,
        # stable foundation while reducing overshoot/ringing risk.
        self.magic_upsampler = MagicKernelSharp2021Upsample(
            in_channels=num_feat,
            alpha=self.upsampler_alpha,
        )

        # First HR conv after upsampling (acts as adaptation layer into HR space).
        self.hr_conv_in = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Lightweight HR refinement head: sequence of ResidualBlocks operating in HR.
        # Corrects subtle ringing/aliasing while preserving structure.
        if self.hr_blocks > 0:
            self.hr_head = nn.Sequential(
                *[ResidualBlock(num_feat) for _ in range(self.hr_blocks)]
            )
        else:
            self.hr_head = nn.Identity()

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
        # Shallow + deep LR features
        x_shallow = self.conv_in(x)
        x_deep = self.body(x_shallow)
        x_fused = self.conv_fuse(x_deep) + x_shallow

        # Magic-based upsampling in feature space
        x_upsampled = self.magic_upsampler(x_fused, scale_factor=self.scale)

        # HR refinement head: small residual stack as corrector
        h = self.hr_conv_in(x_upsampled)
        h = self.hr_head(h)

        # Final RGB reconstruction
        return self.conv_out(h)


# --- Factory Registration for traiNNer-redux: The Redesigned V2 Family ---
#
# Rationale: A streamlined progression of variants that provide meaningful
# differentiation in capacity, speed, and quality while maintaining the
# deployment-optimized architecture that makes ParagonSR2 special.


@ARCH_REGISTRY.register()
def paragonsr2_nano(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-Nano: Ultra-lightweight configuration for rapid prototyping and constrained hardware.

    GOAL: Maximum deployment efficiency with acceptable quality for basic super-resolution.

    WHEN TO USE:
    - Prototyping new training pipelines or loss functions
    - GPU memory constrained environments (~4GB VRAM)
    - Real-time applications where speed is paramount
    - Quick qualitative testing without long training times
    - Mobile/edge deployment where model size matters

    CAPACITY PROFILE:
    - Features: 24 (minimal channel dimension)
    - Groups×Blocks: 2×2 = 4 total transformers (minimal depth)
    - Band Kernel: 9 (smaller context for speed)
    - HR Blocks: 1 (basic refinement)

    EXPECTED RESULTS:
    - Fastest training and inference
    - Basic upscaling quality
    - Suitable for simple degradations
    - Limited capacity for complex perceptual training
    """
    return ParagonSR2(
        scale=scale,
        num_feat=24,
        num_groups=2,
        num_blocks=2,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 9},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 1),
    )


@ARCH_REGISTRY.register()
def paragonsr2_xs(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-XS: Extra-small configuration for balanced speed and quality.

    GOAL: Sweet spot between training efficiency and perceptual capability.

    WHEN TO USE:
    - Initial perceptual training experiments
    - GPU memory limited environments (~6GB VRAM)
    - When you need better quality than Nano but still want speed
    - Testing different loss combinations
    - Quick baseline model training

    CAPACITY PROFILE:
    - Features: 32 (moderate channel dimension)
    - Groups×Blocks: 2×3 = 6 total transformers (light depth)
    - Band Kernel: 11 (balanced spatial context)
    - HR Blocks: 1 (basic refinement)

    EXPECTED RESULTS:
    - Good training speed with improved quality over Nano
    - Can handle basic perceptual losses
    - Better detail reconstruction
    - Suitable for simple perceptual fine-tuning
    """
    return ParagonSR2(
        scale=scale,
        num_feat=32,
        num_groups=2,
        num_blocks=3,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 11},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 1),
    )


@ARCH_REGISTRY.register()
def paragonsr2_s(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-S: Small configuration for mainstream perceptual training.

    GOAL: Optimal balance for high-quality perceptual training on consumer hardware.

    WHEN TO USE:
    - Primary choice for perceptual model training
    - GPU memory: RTX 3060, GTX 1660S (~8GB VRAM)
    - When you want clear visual differences between pretrain/perceptual models
    - Complex loss combinations (DINO, LDL, frequency losses, etc.)
    - Production-quality model training
    - The "goldilocks" variant for most use cases

    CAPACITY PROFILE:
    - Features: 48 (substantial channel dimension)
    - Groups×Blocks: 3×4 = 12 total transformers (moderate depth)
    - Band Kernel: 11 (balanced spatial context)
    - HR Blocks: 2 (enhanced refinement)

    EXPECTED RESULTS:
    - High-quality results with proper training
    - Excellent balance of speed and quality
    - Can handle complex 8+ component loss functions
    - Strong enough for meaningful perceptual improvements
    - Suitable for final production models
    """
    return ParagonSR2(
        scale=scale,
        num_feat=48,
        num_groups=3,
        num_blocks=4,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 11},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 2),
    )


@ARCH_REGISTRY.register()
def paragonsr2_m(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-M: Medium configuration for enhanced quality on better hardware.

    GOAL: High-quality results for users with more VRAM and training time.

    WHEN TO USE:
    - When S variant quality isn't quite sufficient
    - GPU memory: RTX 3070, RTX 4060 Ti (~12GB VRAM)
    - Training models that need stronger global context understanding
    - When artifacts-free results are critical
    - Professional or semi-professional applications

    CAPACITY PROFILE:
    - Features: 64 (large channel dimension)
    - Groups×Blocks: 4×6 = 24 total transformers (substantial depth)
    - Band Kernel: 13 (enhanced spatial context)
    - HR Blocks: 2 (enhanced refinement)

    EXPECTED RESULTS:
    - Superior quality compared to S variant
    - Better handling of complex scenes and textures
    - Improved artifact suppression
    - Can handle the most demanding loss combinations
    - Professional-grade output quality
    """
    return ParagonSR2(
        scale=scale,
        num_feat=64,
        num_groups=4,
        num_blocks=6,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 13},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 2),
    )


@ARCH_REGISTRY.register()
def paragonsr2_l(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-L: Large configuration for near-SOTA quality on high-end hardware.

    GOAL: Maximum quality for users who prioritize results over training efficiency.

    WHEN TO USE:
    - When you need the best possible quality
    - GPU memory: RTX 3080, RTX 4070 Ti, RTX 4080 (~16GB VRAM)
    - Professional video production or critical applications
    - When training time is less important than final quality
    - Research and development projects

    CAPACITY PROFILE:
    - Features: 96 (very large channel dimension)
    - Groups×Blocks: 6×8 = 48 total transformers (deep architecture)
    - Band Kernel: 15 (maximum spatial context)
    - HR Blocks: 3 (maximum refinement)

    EXPECTED RESULTS:
    - Near state-of-the-art quality with proper training
    - Excellent handling of complex degradations
    - Superior artifact-free results
    - Can push perceptual training to its limits
    - Professional/broadcast quality output
    """
    return ParagonSR2(
        scale=scale,
        num_feat=96,
        num_groups=6,
        num_blocks=8,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 15},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 3),
    )


@ARCH_REGISTRY.register()
def paragonsr2_xl(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    V2-XL: Extra-large configuration for research-grade quality.

    GOAL: Maximum capacity for cutting-edge research and benchmarks.

    WHEN TO USE:
    - Research institutions and companies with significant compute resources
    - GPU memory: RTX 4090, A100, H100 (~24GB+ VRAM)
    - Pushing the boundaries of single-image super-resolution
    - When no expense is spared for maximum quality
    - Benchmark submissions and competition entries

    CAPACITY PROFILE:
    - Features: 128 (research-grade channel dimension)
    - Groups×Blocks: 8×10 = 80 total transformers (very deep)
    - Band Kernel: 15 (maximum spatial context)
    - HR Blocks: 3 (maximum refinement)

    EXPECTED RESULTS:
    - State-of-the-art quality potential
    - Maximum capacity for handling complex degradations
    - Can learn the most subtle texture details
    - Best-in-class perceptual quality
    - Research/benchmark-grade results
    """
    return ParagonSR2(
        scale=scale,
        num_feat=128,
        num_groups=8,
        num_blocks=10,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 15},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 3),
    )
