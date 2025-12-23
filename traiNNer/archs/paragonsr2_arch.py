#!/usr/bin/env python3
"""
ParagonSR2
==========

A dual-path, convolution-first SISR architecture with selective local attention and deployment-first design.

Key principles:
- Dual-path reconstruction (fixed classical base + learned residual detail)
- Variant specialization (Realtime / Stream / Photo)
- Convolution-first design with selective attention
- Export- and deployment-friendly (ONNX / TensorRT / FP16)
- Stable eager execution
- Optional PyTorch-only inference optimizations

Author: Philip Hofmann
License: MIT
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import checkpoint

from traiNNer.utils.registry import ARCH_REGISTRY

# ============================================================================
# 1. CLASSICAL BASE UPSAMPLER (MAGIC KERNEL SHARP 2021)
# ============================================================================


def get_magic_kernel_weights():
    """
    Low-pass reconstruction kernel from Costella's Magic Kernel.
    """
    return torch.tensor([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])


def get_magic_sharp_2021_kernel_weights():
    """
    Sharpening kernel from Magic Kernel Sharp 2021.
    """
    return torch.tensor([-1 / 32, 0, 9 / 32, 16 / 32, 9 / 32, 0, -1 / 32])


class SeparableConv(nn.Module):
    """
    Fixed separable convolution for classical reconstruction kernels.

    These filters are frozen by design and must never be trained.
    """

    def __init__(self, channels: int, kernel: torch.Tensor) -> None:
        super().__init__()
        k = len(kernel)
        self.register_buffer("kernel", kernel)

        self.h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=channels,
            bias=False,
        )
        self.v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            groups=channels,
            bias=False,
        )

        with torch.no_grad():
            self.h.weight.copy_(kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1))
            self.v.weight.copy_(kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1))

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.h(x))


class MagicKernelSharp2021Upsample(nn.Module):
    """
    Fixed classical upsampler based on Magic Kernel Sharp 2021.

    Provides a strong low-frequency base reconstruction that the neural
    network refines with learned high-frequency detail.
    """

    def __init__(self, in_ch: int, scale: int, alpha: float) -> None:
        super().__init__()
        self.scale = scale
        self.alpha = alpha

        self.sharp = SeparableConv(in_ch, get_magic_sharp_2021_kernel_weights())
        self.blur = SeparableConv(in_ch, get_magic_kernel_weights())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional pre-sharpening
        if self.alpha > 0:
            x = x + self.alpha * (self.sharp(x) - x)

        # Nearest-neighbor upsampling
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")

        # Reconstruction blur
        return self.blur(x)


# ============================================================================
# 2. NORMALIZATION & RESIDUAL SCALING
# ============================================================================


class RMSNorm(nn.Module):
    """
    Spatial RMS normalization.

    More stable than BatchNorm for SR and safe in FP16.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return self.scale * x / rms + self.bias


class LayerScale(nn.Module):
    """
    Residual scaling for training stability.
    """

    def __init__(self, channels: int, init: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ============================================================================
# 3. CORE BLOCKS
# ============================================================================


class WindowAttention(nn.Module):
    """
    Simplified Window Attention (Swin-style).

    Partitions the input into non-overlapping windows and computes attention
    locally within each window. Supports window shifting for cross-window connections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        attention_mode: str = "sdpa",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.attention_mode = attention_mode
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # Pad features to multiples of window size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        # Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        # (B, Hp, Wp, C) -> (B, h_win, w_win, ws, ws, C)
        x_windows = x.view(
            B,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
            C,
        )
        x_windows = x_windows.permute(
            0, 1, 3, 2, 4, 5
        ).contiguous()  # (B, h_win, w_win, ws, ws, C)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # (num_windows, ws*ws, C)

        # Attention
        qkv = self.qkv(x_windows)
        q, k, v = qkv.chunk(3, dim=-1)  # (num_windows, N, C)

        # Multi-head split
        q = q.view(
            -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
        ).transpose(1, 2)
        k = k.view(
            -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
        ).transpose(1, 2)
        v = v.view(
            -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
        ).transpose(1, 2)

        if self.attention_mode == "flex":
            # We rely on dynamic import or assuming it's available if this mode is picked
            try:
                from torch.nn.attention import flex_attention
            except ImportError:
                raise RuntimeError(
                    "FlexAttention requested but not available in this PyTorch build."
                )
            x_windows = flex_attention(q, k, v)
        else:
            # Standard SDPA
            x_windows = F.scaled_dot_product_attention(q, k, v)

        x_windows = (
            x_windows.transpose(1, 2)
            .contiguous()
            .view(-1, self.window_size * self.window_size, C)
        )
        x_windows = self.proj(x_windows)

        # Reverse Partition
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = x_windows.view(
            B,
            Hp // self.window_size,
            Wp // self.window_size,
            self.window_size,
            self.window_size,
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        # Reverse Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        return x


class NanoBlock(nn.Module):
    """
    Ultra-lightweight block for the Realtime variant.

    Optimized for throughput:
    - No gating
    - No attention
    - Minimal receptive field
    """

    def __init__(self, dim: int, expansion: float = 2.0, **kwargs) -> None:
        super().__init__()
        hidden = int(dim * expansion)

        self.conv1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.gelu(self.dw(self.conv1(x)))
        x = self.conv2(x)
        return x + res


class StreamBlock(nn.Module):
    """
    Video-oriented block.

    Designed to suppress compression artifacts using:
    - Multi-rate depthwise context
    - Simple gating
    - Fully convolutional operations
    """

    def __init__(self, dim: int, expansion: float = 2.0, **kwargs) -> None:
        super().__init__()
        hidden = int(dim * expansion)

        self.dw1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim)

        self.fuse = nn.Conv2d(dim * 2, dim, 1)

        self.proj = nn.Conv2d(dim, hidden * 2, 1)
        self.gate = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        x = torch.cat([self.dw1(x), self.dw3(x)], dim=1)
        x = self.fuse(x)

        x = self.proj(x)
        x = self.gate(x)
        a, b = x.chunk(2, dim=1)
        x = a * b

        x = self.out(x)
        return x + res


class PhotoBlock(nn.Module):
    """
    Photo-oriented block.

    Strong convolutional mixing with optional attention for
    long-range structural consistency.

    Attention can be:
    - Disabled (export-safe)
    - SDPA (default)
    - FlexAttention (PyTorch-only inference)
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        attention_mode: str | None = "sdpa",
        export_safe: bool = False,
        window_size: int = 16,
        shift_size: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        hidden = int(dim * expansion)

        self.norm = RMSNorm(dim)
        self.conv1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 1)
        self.scale = LayerScale(dim)

        self.attention_mode = attention_mode
        self.export_safe = export_safe

        if attention_mode is not None and not export_safe:
            self.attn_norm = RMSNorm(dim)
            self.attn = WindowAttention(
                dim,
                num_heads=4,
                window_size=window_size,
                shift_size=shift_size,
                attention_mode=attention_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        # Convolutional path
        x = self.norm(x)
        x = self.conv1(x)
        x = F.gelu(self.dw(x))
        x = self.conv2(x)
        x = res + self.scale(x)

        # Optional attention
        if self.attention_mode is not None and not self.export_safe:
            # WindowAttention expects (B, H, W, C) for easier manipulation
            res_attn = x
            x = self.attn_norm(x).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            x = self.attn(x)
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            x = res_attn + self.scale(x)

        return x


# ============================================================================
# 4. RESIDUAL GROUP
# ============================================================================


class ResidualGroup(nn.Module):
    """
    Group of blocks with optional gradient checkpointing.
    """

    def __init__(self, blocks: list[nn.Module], checkpointing: bool = False) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*blocks)
        self.checkpointing = checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpointing and x.requires_grad:
            for b in self.blocks:
                x = checkpoint.checkpoint(b, x, use_reentrant=False)
            return x
        return self.blocks(x)


# ============================================================================
# 5. MAIN NETWORK
# ============================================================================


@ARCH_REGISTRY.register()
class ParagonSR2(nn.Module):
    """
    ParagonSR2 main SISR generator.
    """

    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 4,
        num_blocks: int = 4,
        variant: str = "photo",
        detail_gain: float = 0.1,
        upsampler_alpha: float = 0.5,
        use_checkpointing: bool = False,
        attention_mode: str | None = "sdpa",
        export_safe: bool = False,
        window_size: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()

        self.base = MagicKernelSharp2021Upsample(in_chans, scale, upsampler_alpha)

        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, padding=1)

        # Helper to construct blocks with alternating shift
        def build_blocks(group_idx: int):
            # Shifted windows are optional and mainly for photo consistency.
            # They can be disabled entirely for maximum simplicity.
            blocks = []
            for i in range(num_blocks):
                # Calculate global block index to alternate shifts
                block_idx = group_idx * num_blocks + i
                shift_size = (window_size // 2) if (block_idx % 2 != 0) else 0

                if variant == "realtime":
                    blocks.append(NanoBlock(num_feat))
                elif variant == "stream":
                    blocks.append(StreamBlock(num_feat))
                elif variant == "photo":
                    blocks.append(
                        PhotoBlock(
                            num_feat,
                            attention_mode=attention_mode,
                            export_safe=export_safe,
                            window_size=window_size,
                            shift_size=shift_size,
                        )
                    )
                else:
                    raise ValueError(f"Unknown variant: {variant}")
            return blocks

        self.body = nn.Sequential(
            *[
                ResidualGroup(
                    build_blocks(g),
                    checkpointing=use_checkpointing,
                )
                for g in range(num_groups)
            ]
        )

        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, padding=1)

        self.up = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, padding=1)
        self.detail_gain = nn.Parameter(torch.tensor(detail_gain))

    def forward(
        self, x: torch.Tensor, feature_tap: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        base = self.base(x)

        x = self.conv_in(x)
        feat = x if feature_tap else None

        x = self.body(x)
        x = self.conv_mid(x)
        x = self.up(x)

        detail = self.conv_out(x) * self.detail_gain
        out = base + detail

        if feature_tap:
            return out, feat
        return out


# ============================================================================
# 6. FACTORY FUNCTIONS
# ============================================================================


@ARCH_REGISTRY.register()
def paragonsr2_realtime(scale=4, **kw):
    return ParagonSR2(
        scale=scale,
        num_feat=16,
        num_groups=1,
        num_blocks=3,
        variant="realtime",
        detail_gain=kw.pop("detail_gain", 0.05),
        upsampler_alpha=kw.pop("upsampler_alpha", 0.3),
        **kw,
    )


@ARCH_REGISTRY.register()
def paragonsr2_stream(scale=4, **kw):
    return ParagonSR2(
        scale=scale,
        num_feat=32,
        num_groups=2,
        num_blocks=3,
        variant="stream",
        detail_gain=kw.pop("detail_gain", 0.1),
        upsampler_alpha=kw.pop("upsampler_alpha", 0.0),
        **kw,
    )


@ARCH_REGISTRY.register()
def paragonsr2_photo(scale=4, **kw):
    return ParagonSR2(
        scale=scale,
        num_feat=64,
        num_groups=4,
        num_blocks=4,
        variant="photo",
        detail_gain=kw.pop("detail_gain", 0.1),
        upsampler_alpha=kw.pop("upsampler_alpha", 0.4),
        attention_mode=kw.pop("attention_mode", "sdpa"),
        export_safe=kw.pop("export_safe", False),
        window_size=kw.pop("window_size", 16),
        **kw,
    )
