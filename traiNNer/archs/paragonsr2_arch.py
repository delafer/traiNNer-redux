#!/usr/bin/env python3
"""
ParagonSR2 Hybrid - Efficient Super-Resolution with Dual-Path Architecture

Author: Philip Hofmann
License: MIT
Repository: https://github.com/Phhofm/traiNNer-redux

═══════════════════════════════════════════════════════════════════════════════
DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════════

This architecture bridges the gap between lightweight CNNs and heavy Transformers.
It employs a Dual-Path design to balance structural integrity with high-frequency
texture restoration.

Key Innovation: Dual-Path Architecture
--------------------------------------
Path A (Detail):  LR → Deep Body (LR Space) → PixelShuffle → Detail Residual
Path B (Base):    LR → MagicKernelSharp → Base Upscale
Output = Base + (Detail * ContentGain)

This design provides:
1. Efficiency: All heavy processing occurs in Low-Resolution (LR) space.
2. Stability: MagicKernel anchor prevents structural distortion.
3. Versatility: Specialized block types for Video vs. Photo restoration.
4. Deployment: ONNX/TensorRT friendly (static graphs preferred).

Variant Strategy:
-----------------
1. Realtime (Nano): Ultra-fast (MBConv), 16ch, for 60fps video/anime.
2. Stream (Tiny):   De-blocking focused (GateBlock), 32ch, wide receptive field.
3. Photo (Base):    Balanced (ParagonBlock), 64ch, for general photography.
4. Pro (Large):     High Fidelity (ParagonBlock+Attn), 96ch, for archival/print.
"""

from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F
import torch.onnx
from torch import nn
from torch.utils import checkpoint

from traiNNer.utils.registry import ARCH_REGISTRY

# ═════════════════════════════════════════════════════════════════════════════
# 1. HELPER UTILS & MAGIC KERNEL
# ═════════════════════════════════════════════════════════════════════════════


def get_magic_kernel_weights() -> torch.Tensor:
    """B-spline kernel for smooth upsampling (Magic Kernel)."""
    return torch.tensor([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])


def get_magic_sharp_2021_kernel_weights() -> torch.Tensor:
    """Sharpening kernel for detail enhancement (MagicKernelSharp2021)."""
    return torch.tensor([-1 / 32, 0, 9 / 32, 16 / 32, 9 / 32, 0, -1 / 32])


class SeparableConv(nn.Module):
    """
    Separable 1D convolution (horizontal then vertical).
    Fixed weights (no gradients) - used for MagicKernel Upsampling.
    """

    def __init__(self, in_channels: int, kernel: torch.Tensor) -> None:
        super().__init__()
        kernel_size = len(kernel)
        # Horizontal convolution (1 x K)
        self.conv_h = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=in_channels,
            bias=False,
        )
        # Vertical convolution (K x 1)
        self.conv_v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=in_channels,
            bias=False,
        )
        # Initialize and freeze weights
        with torch.no_grad():
            reshaped = kernel.view(1, 1, 1, -1).repeat(in_channels, 1, 1, 1)
            self.conv_h.weight.copy_(reshaped)
            reshaped = kernel.view(1, 1, -1, 1).repeat(in_channels, 1, 1, 1)
            self.conv_v.weight.copy_(reshaped)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_v(self.conv_h(x))


class MagicKernelSharp2021Upsample(nn.Module):
    """
    MagicKernelSharp2021 upsampler (classical method).
    Provides the structural 'Base' for the dual-path architecture.
    """

    def __init__(self, in_channels: int, scale: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.scale = scale
        self.alpha = max(0.0, min(alpha, 1.0))

        sharp_kernel = get_magic_sharp_2021_kernel_weights()
        self.sharpen = SeparableConv(in_channels, sharp_kernel)

        resample_kernel = get_magic_kernel_weights()
        self.resample_conv = SeparableConv(in_channels, resample_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional sharpening (skip if alpha is 0 for pure reconstruction)
        if self.alpha > 0.0:
            x_sharp = self.sharpen(x)
            x = x + self.alpha * (x_sharp - x)

        # Nearest-neighbor upsampling + B-Spline Blur
        x_upsampled = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.resample_conv(x_upsampled)


def icnr_init(conv_weight, scale=4, init=nn.init.kaiming_normal_) -> None:
    """ICNR initialization to prevent checkerboard artifacts."""
    ni, nf, h, w = conv_weight.shape
    if ni < scale**2:
        return  # Skip if channels are insufficient

    output_shape = ni // (scale**2)
    kernel = torch.zeros([output_shape, nf, h, w]).transpose(0, 1)
    init(kernel)
    kernel = kernel.transpose(0, 1)
    kernel = kernel.repeat(1, 1, scale, scale)
    conv_weight.data.copy_(kernel.reshape(ni, nf, h, w))


class PixelShufflePack(nn.Module):
    """Learned upsampling via sub-pixel convolution."""

    def __init__(
        self, in_channels: int, out_channels: int, scale: int, upsample_kernel: int = 3
    ) -> None:
        super().__init__()
        self.up_conv = nn.Conv2d(
            in_channels,
            out_channels * (scale**2),
            upsample_kernel,
            padding=upsample_kernel // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        icnr_init(self.up_conv.weight, scale=scale)
        if self.up_conv.bias is not None:
            self.up_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.up_conv(x))


class ProgressivePixelShufflePack(nn.Module):
    """
    Progressive upsampling for larger scales (4x, 8x).

    Uses multiple 2x stages with intermediate convolutions for cleaner results:
    - scale=2 or 3: Single stage (falls back to PixelShufflePack)
    - scale=4: Two 2x stages with intermediate refinement
    - scale=8: Three 2x stages with intermediate refinement

    Benefits:
    - Reduced checkerboard artifacts at large scales
    - Better gradient flow during training
    - More natural texture reconstruction
    """

    def __init__(
        self, in_channels: int, out_channels: int, scale: int, upsample_kernel: int = 3
    ) -> None:
        super().__init__()
        self.scale = scale

        if scale in {2, 3}:
            # Single stage for small scales
            self.stages = nn.ModuleList(
                [PixelShufflePack(in_channels, out_channels, scale, upsample_kernel)]
            )
        elif scale == 4:
            # Two 2x stages: in_ch -> in_ch (2x) -> out_ch (2x)
            self.stages = nn.ModuleList(
                [
                    PixelShufflePack(in_channels, in_channels, 2, upsample_kernel),
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                    nn.LeakyReLU(0.1, True),
                    PixelShufflePack(in_channels, out_channels, 2, upsample_kernel),
                ]
            )
        elif scale == 8:
            # Three 2x stages: in_ch -> in_ch (2x) -> in_ch (2x) -> out_ch (2x)
            self.stages = nn.ModuleList(
                [
                    PixelShufflePack(in_channels, in_channels, 2, upsample_kernel),
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                    nn.LeakyReLU(0.1, True),
                    PixelShufflePack(in_channels, in_channels, 2, upsample_kernel),
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                    nn.LeakyReLU(0.1, True),
                    PixelShufflePack(in_channels, out_channels, 2, upsample_kernel),
                ]
            )
        else:
            raise ValueError(
                f"ProgressivePixelShufflePack: scale must be 2, 3, 4, or 8. Got {scale}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 2. CORE LAYERS (Normalization, Attention, Processors)
# ═════════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Faster than LayerNorm/GroupNorm, ideal for SISR inference.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.offset = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed + self.offset


class LayerScale(nn.Module):
    """Scaling layer for residual path stability."""

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(init_values)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ContentAwareDetailProcessor(nn.Module):
    """
    Content-Aware Gating for intelligent detail modulation.

    Analyzes DEEP FEATURES (not LR input) to distinguish between:
    - Smooth areas (walls, sky) -> suppress detail gain to prevent noise
    - Textured areas (grass, fabric) -> boost detail gain for restoration

    Key Insight: Deep features contain "hallucinated" texture patterns that
    allow better texture vs. noise discrimination than the blurry LR input.
    """

    def __init__(self, num_feat: int) -> None:
        super().__init__()
        # Determine hidden channel size (min 8 to prevent collapse in Nano)
        hidden = max(8, num_feat // 4)

        # Analyzer operates on deep features (num_feat channels)
        self.analyzer = nn.Sequential(
            nn.Conv2d(num_feat, hidden, 3, 1, 1),  # Project from feature space
            nn.LeakyReLU(0.1, True),
            # Large receptive field to detect structures vs noise
            nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output range: [0.0, 2.0]
        # Default (0.5 from Sigmoid) maps to 1.0 gain (neutral)
        return self.analyzer(x) * 2.0


class LocalWindowAttention(nn.Module):
    """
    Shifted Window Self-Attention (Swin-style) for 'Pro' and 'Photo' variants.

    Captures global consistency within patches while using shifted windows
    to create cross-window connections and prevent visible boundary artifacts.

    Uses F.scaled_dot_product_attention (FlashAttention) for maximum efficiency.

    Args:
        dim: Number of input channels
        window_size: Size of attention window (default 32)
        num_heads: Number of attention heads
        shift_size: Shift offset for creating overlapping receptive fields.
                   Set to window_size//2 for odd blocks, 0 for even blocks.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 32,
        num_heads: int = 4,
        shift_size: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        # Static flag to print attention mode once
        if not hasattr(LocalWindowAttention, "_printed_mode"):
            LocalWindowAttention._printed_mode = False

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Shortcut for small images
        if H <= self.window_size and W <= self.window_size:
            return self._attn(x)

        # Pad to multiple of window size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        # Always pad to ensure graph continuity in ONNX (even if 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        # ─── SHIFTED WINDOW (Swin-style) ───
        # Apply cyclic shift to create cross-window connections
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # Window Partition (Split into H and W steps to avoid 6D permute - TRT Optimization)
        grid_h = Hp // self.window_size
        grid_w = Wp // self.window_size

        # Step 1: Partition Height
        # (B, C, H, W) -> (B, C, grid_h, win, W)
        x = x.view(B, C, grid_h, self.window_size, Wp)
        # (B, C, grid_h, win, W) -> (B, grid_h, C, win, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, self.window_size, Wp)

        # Step 2: Partition Width
        # (B*grid_h, C, win, W) -> (B*grid_h, C, win, grid_w, win)
        x = x.view(-1, C, self.window_size, grid_w, self.window_size)
        # (B*grid_h, C, win, grid_w, win) -> (B*grid_h, grid_w, C, win, win)
        x = x.permute(0, 3, 1, 2, 4).reshape(-1, C, self.window_size, self.window_size)

        # Attention
        x = self._attn(x)

        # Window Reverse
        # Step 1: Reverse Width
        # (B*grid_h*grid_w, C, win, win) -> (B*grid_h, grid_w, C, win, win)
        x = x.view(-1, grid_w, C, self.window_size, self.window_size)
        # (B*grid_h, grid_w, C, win, win) -> (B*grid_h, C, win, grid_w, win)
        x = x.permute(0, 2, 3, 1, 4).reshape(-1, C, self.window_size, Wp)

        # Step 2: Reverse Height
        # (B*grid_h, C, win, W) -> (B, grid_h, C, win, W)
        x = x.view(B, grid_h, C, self.window_size, Wp)
        # (B, grid_h, C, win, W) -> (B, C, grid_h, win, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C, Hp, Wp)

        # ─── REVERSE SHIFT ───
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # Always slice to restore original size (handling the pad=0 case automatically)
        x = x[:, :, :H, :W]

        return x

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Reshape to (B, 3, Heads, HeadDim, Pixels)
        # Note: Pixels is sequence length
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, -1)

        # q, k, v shapes: (B, Heads, HeadDim, Pixels)
        q, k, v = qkv.unbind(1)

        # ---------------------------------------------------------------------
        # ONNX Export Path: Use Standard Math (MatMul + Softmax)
        # ---------------------------------------------------------------------
        # SDPA is often not supported or exports to complex plugins in older opsets.
        # We manually implement Attention so it fuses into TRT's native kernels.
        if torch.onnx.is_in_onnx_export():
            if not LocalWindowAttention._printed_mode:
                print("[ParagonSR2] Export Mode Detected: Using Manual Attention.")
                LocalWindowAttention._printed_mode = True

            # Prepare for BMM: (B * Heads, HeadDim, Pixels)
            q = q.reshape(B * self.num_heads, -1, H * W)
            k = k.reshape(B * self.num_heads, -1, H * W)
            v = v.reshape(B * self.num_heads, -1, H * W)

            # Attention Map: (B*Heads, Pixels, Pixels)
            # q.transpose(-2, -1) -> (..., Pixels, HeadDim)
            # k -> (..., HeadDim, Pixels)
            dots = torch.bmm(q.transpose(-2, -1), k) * self.scale
            attn = F.softmax(dots, dim=-1)

            # Weighted Sum: (B*Heads, HeadDim, Pixels)
            # v -> (..., HeadDim, Pixels)
            # attn.transpose(-2, -1) -> (..., Pixels, Pixels) (Simulates V @ Attn_T)
            out = torch.bmm(v, attn.transpose(-2, -1))

            # Reshape back to (B, C, H, W)
            # (B*Heads, HeadDim, Pixels) -> (B, Heads, HeadDim, Pixels)
            out = out.reshape(B, self.num_heads, -1, H * W)
            out = out.reshape(B, C, H, W)
            return self.proj(out)

        # ---------------------------------------------------------------------
        # Training/Inference Path: Use FlashAttention (SDPA)
        # ---------------------------------------------------------------------
        if not LocalWindowAttention._printed_mode:
            print("[ParagonSR2] Training Mode: Using FlashAttention (SDPA).")
            LocalWindowAttention._printed_mode = True

        # Permute for SDPA: (B, Heads, Pixels, HeadDim)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Use efficient SDPA (FlashAttention/MemEfficient)
        # Automatically handles scaling (1/sqrt(head_dim)) if scale is None, matches our self.scale
        out = F.scaled_dot_product_attention(q, k, v)

        # Reshape back: (B, Heads, Pixels, HeadDim) -> (B, Heads, HeadDim, Pixels)
        out = out.transpose(-2, -1)

        # Restore shape: (B, Heads, HeadDim, Pixels) -> (B, C, H, W)
        out = out.reshape(B, C, H, W)
        return self.proj(out)


class InceptionDWConv2d(nn.Module):
    """
    Multi-scale context gathering module.
    Splits channels and applies different kernel sizes.
    """

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,  # 11 for Photo, 21 for Stream, 17 for Pro
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


# ═════════════════════════════════════════════════════════════════════════════
# 3. BLOCK ARCHITECTURES (The Variant Engines)
# ═════════════════════════════════════════════════════════════════════════════


class NanoBlock(nn.Module):
    """
    [Optimized for Realtime]
    MBConv-style block. Pure speed.
    Used in 'paragonsr2_realtime'.
    """

    def __init__(self, dim: int, expansion: float = 2.0, **kwargs) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.net = nn.Sequential(
            # 1. Expand
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.Hardswish(),  # Faster on mobile/RT
            # 2. Depthwise Context
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.Hardswish(),
            # 3. Project
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SimpleGateBlock(nn.Module):
    """
    [Optimized for Stream/Video]
    Static gating block. Replaces Transformer/Attention for robust video handling.
    Used in 'paragonsr2_stream'.
    """

    def __init__(self, dim: int, expansion: float = 2.0, **kwargs) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion)

        # Wide-context Inception for de-blocking (optional, or just use larger receptive field separately)
        self.context = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        # Simple Gated FFN
        self.proj_in = nn.Conv2d(dim, hidden_dim * 2, 1)
        self.dw_gate = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2, 3, 1, 1, groups=hidden_dim * 2
        )
        self.proj_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.context(x)

        # Gating mechanism
        x = self.proj_in(x)
        x = self.dw_gate(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2  # Element-wise self-attention
        x = self.proj_out(x)

        return x + x_in


class ParagonBlock(nn.Module):
    """
    [Optimized for Photo/Pro]
    The full-featured block with Context, Gating, and optional Shifted Window Attention.

    Uses Swin-style alternating shift pattern when attention is enabled:
    - Even indexed blocks: Regular windows (shift_size=0)
    - Odd indexed blocks: Shifted windows (shift_size=window_size//2)
    """

    def __init__(
        self,
        dim: int,
        ffn_expansion: float = 2.0,
        use_attention: bool = False,
        band_kernel_size: int = 11,
        block_idx: int = 0,  # Used for alternating shift pattern
        window_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        # 1. Multi-scale Context
        self.context = InceptionDWConv2d(dim, band_kernel_size=band_kernel_size)
        self.ls1 = LayerScale(dim)

        # 2. Mixing / Attention
        hidden_dim = int(dim * ffn_expansion)
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        self.dw_mixer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )

        # Shifted Window Attention with alternating pattern
        if use_attention:
            # Odd blocks get shifted windows for cross-window connections
            shift_size = (window_size // 2) if (block_idx % 2 == 1) else 0
            self.attn = LocalWindowAttention(
                hidden_dim, window_size=window_size, shift_size=shift_size
            )
        else:
            self.attn = nn.Identity()

        # Gating (via Cheap Channel Mod)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.project_out = nn.Conv2d(hidden_dim, dim, 1)
        self.ls2 = LayerScale(dim)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.context(x)
        x = res + self.ls1(x)

        res = x
        x = self.norm(x)
        x = self.project_in(x)
        x = self.dw_mixer(x)
        x = self.attn(x)
        x = x * self.gate(x)  # Channel Modulation
        x = self.project_out(x)
        x = res + self.ls2(x)

        return x


class ResidualGroup(nn.Module):
    """Generic Residual Group that selects the correct block type."""

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        block_type: str = "paragon",
        group_idx: int = 0,  # Used for global block indexing
        use_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing

        BlockClass: type[nn.Module]
        if block_type == "nano":
            BlockClass = NanoBlock
        elif block_type == "gate":
            BlockClass = SimpleGateBlock
        else:
            BlockClass = ParagonBlock

        # Pass block_idx for alternating shift patterns in attention
        # Global block index = group_idx * num_blocks + local_idx
        self.blocks = nn.Sequential(
            *[
                BlockClass(dim, block_idx=group_idx * num_blocks + i, **kwargs)
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and x.requires_grad:
            res = x
            for block in self.blocks:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            return x + res
        else:
            return self.blocks(x) + x


# ═════════════════════════════════════════════════════════════════════════════
# 4. MAIN NETWORK ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════


@ARCH_REGISTRY.register()
class ParagonSR2(nn.Module):
    """
    ParagonSR2 Main Class.
    Constructs the dual-path network based on the provided configuration.
    """

    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 4,
        num_blocks: int = 6,
        ffn_expansion: float = 2.0,
        upsampler_alpha: float = 0.5,
        detail_gain: float = 0.1,
        use_content_aware: bool = True,
        block_type: str = "paragon",  # 'nano', 'gate', 'paragon'
        block_kwargs: dict | None = None,
        use_channels_last: bool = True,
        use_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validation
        if scale not in [2, 3, 4, 8]:
            raise ValueError(f"Scale must be 2, 3, 4, or 8. Got {scale}.")

        self.scale = scale
        self.use_channels_last = use_channels_last and torch.cuda.is_available()
        self.use_content_aware = use_content_aware

        if block_kwargs is None:
            block_kwargs = {}

        # ─── PATH A: BASE (MagicKernel) ───
        self.base_upsampler = MagicKernelSharp2021Upsample(
            in_channels=in_chans, scale=scale, alpha=upsampler_alpha
        )

        # ─── PATH B: DETAIL (Deep Body) ───

        # 1. Feature Extraction
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1)

        # 2. Deep Processing Body (with group indices for alternating attention)
        self.body = nn.Sequential(
            *[
                ResidualGroup(
                    dim=num_feat,
                    num_blocks=num_blocks,
                    block_type=block_type,
                    group_idx=i,  # Pass group index for global block indexing
                    ffn_expansion=ffn_expansion,
                    use_checkpointing=use_checkpointing,
                    **block_kwargs,
                )
                for i in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 3. Upsampling (Progressive for scale >= 4)
        if scale >= 4:
            self.detail_upsampler = ProgressivePixelShufflePack(
                num_feat, num_feat, scale=scale
            )
        else:
            self.detail_upsampler = PixelShufflePack(num_feat, num_feat, scale=scale)

        # 4. Detail Projection
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

        # ─── LEARNABLE DETAIL GAIN ───
        # Instead of baking gain into conv weights, use a learnable parameter
        # This gives the optimizer a direct "volume knob" for detail magnitude
        self.global_detail_gain = nn.Parameter(torch.tensor(detail_gain))

        # ─── CONTENT-AWARE PROCESSING ───
        self.content_processor = (
            ContentAwareDetailProcessor(num_feat) if use_content_aware else None
        )

        if self.use_channels_last:
            self._to_channels_last()

    def _to_channels_last(self) -> None:
        """Optimization for Tensor Cores."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight") and module.weight.requires_grad:
                    with torch.no_grad():
                        module.weight.data = module.weight.contiguous(
                            memory_format=torch.channels_last
                        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channels Last optimization
        if self.use_channels_last and x.is_cuda:
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)

        # 1. Base Path (Structural Anchor)
        x_base = self.base_upsampler(x)

        # 2. Detail Path (Texture Recovery)
        out = self.conv_in(x)
        out = self.body(out)
        deep_features = self.conv_fuse(out) + out

        # 3. Content-Aware Modulation (on DEEP FEATURES, not LR input)
        # This allows better texture vs. noise discrimination
        if self.content_processor is not None:
            # Analyze deep features to determine where to add details
            gain_map = self.content_processor(deep_features)
            # Apply gain in feature space (before upsampling)
            deep_features = deep_features * gain_map

        # 4. Upsample and project to RGB
        out = self.detail_upsampler(deep_features)
        x_detail = self.conv_out(out)

        # 5. Apply learnable global detail gain
        x_detail = x_detail * self.global_detail_gain

        # 6. Fusion
        return x_base + x_detail


# ═════════════════════════════════════════════════════════════════════════════
# 5. PRODUCT VARIANTS (Factory Functions)
# ═════════════════════════════════════════════════════════════════════════════


@ARCH_REGISTRY.register()
def paragonsr2_realtime(
    scale: int = 4, upsampler_alpha: float = 0.35, **kwargs
) -> ParagonSR2:
    """
    [Realtime Edition] - The 'Nano'
    Target: 1080p -> 4K Gaming/Anime @ 60fps.
    Hardware: iGPU, Mobile, Mid-range.
    Tech: 16ch, MBConv (NanoBlock), No Attention.
    """
    return ParagonSR2(
        scale=scale,
        num_feat=16,
        num_groups=1,
        num_blocks=3,
        ffn_expansion=1.5,
        upsampler_alpha=upsampler_alpha,
        detail_gain=kwargs.pop("detail_gain", 0.05),
        use_content_aware=kwargs.pop("use_content_aware", False),  # Zero overhead
        block_type="nano",
        **kwargs,
    )


@ARCH_REGISTRY.register()
def paragonsr2_stream(
    scale: int = 4, upsampler_alpha: float = 0.0, **kwargs
) -> ParagonSR2:
    """
    [Stream Edition] - The 'Tiny'
    Target: Compressed Video (YouTube/Twitch).
    Hardware: RTX 3050 / Laptops.
    Tech: 32ch, GateBlock, Wide Receptive Field (21), Alpha 0.
    """
    return ParagonSR2(
        scale=scale,
        num_feat=32,
        num_groups=2,
        num_blocks=3,
        upsampler_alpha=upsampler_alpha,
        detail_gain=kwargs.pop("detail_gain", 0.1),
        use_content_aware=kwargs.pop("use_content_aware", True),
        block_type="gate",
        block_kwargs={"band_kernel_size": 21},
        **kwargs,
    )


@ARCH_REGISTRY.register()
def paragonsr2_photo(
    scale: int = 4, upsampler_alpha: float = 0.4, **kwargs
) -> ParagonSR2:
    """
    [Photo Edition] - The 'Base'
    Target: General Photography / Wallpapers.
    Hardware: RTX 3060 / 4060.
    Tech: 64ch, ParagonBlock, Shifted Window Attention, Content-Aware.
    """
    return ParagonSR2(
        scale=scale,
        num_feat=64,
        num_groups=4,
        num_blocks=4,
        upsampler_alpha=upsampler_alpha,
        detail_gain=kwargs.pop("detail_gain", 0.1),
        use_content_aware=kwargs.pop("use_content_aware", True),
        block_type="paragon",
        block_kwargs={"band_kernel_size": 11, "use_attention": True},
        **kwargs,
    )


@ARCH_REGISTRY.register()
def paragonsr2_pro(
    scale: int = 4, upsampler_alpha: float = 0.6, **kwargs
) -> ParagonSR2:
    """
    [Pro Edition] - The 'Large'
    Target: Archival Restoration / Benchmark Fidelity.
    Hardware: RTX 3080 / 4080 / 4090.
    Tech: 96ch, Deep Context (17), Shifted Window Attention, Aggressive Detail.

    Note: For benchmark metrics (PSNR), set upsampler_alpha=0.0 in training.
    """
    return ParagonSR2(
        scale=scale,
        num_feat=96,
        num_groups=6,
        num_blocks=6,
        upsampler_alpha=upsampler_alpha,
        detail_gain=kwargs.pop("detail_gain", 0.15),
        use_content_aware=kwargs.pop("use_content_aware", True),
        block_type="paragon",
        block_kwargs={"band_kernel_size": 17, "use_attention": True},
        **kwargs,
    )
