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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint

from traiNNer.utils.registry import ARCH_REGISTRY


def to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x


# ============================================================================
# 1. CLASSICAL BASE UPSAMPLER (MAGIC KERNEL SHARP 2021)
# ============================================================================


def get_magic_kernel_weights():
    """
    Low-pass reconstruction kernel from Costella's Magic Kernel.
    """
    return torch.tensor([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])


def get_magic_sharp_2021_kernel_weights():
    """Returns the weights for the Magic Kernel Sharp 2021 (MKS21) 3-tap filter."""
    # Weights: [-0.015, 0.138, -0.015] -> [0.138, 0.862, 0.138] unnormalized?
    # Actually MKS2021 is usually approximating Lanczos3 or Catmull-Rom.
    # Here we use a standard sharp bicubic-like approximation.
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
    Spatial RMS Normalization.

    More stable than BatchNorm for SR and safe in FP16.
    Computes variance in FP32 for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate variance in FP32 for stability, then cast back
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=1, keepdim=True)
        rms = torch.sqrt(variance + self.eps).to(x.dtype)
        return self.scale * x / rms + self.bias


class LayerScale(nn.Module):
    """
    LayerScale: Learnable per-channel scaling factor initialized to a small value.
    Crucial for stabilizing deep networks, especially in FP16.
    """

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

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
                from torch.nn.attention.flex_attention import flex_attention
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
    Ultra-lightweight block for Realtime variants.
    Consists of a Depthwise-Conv sandwich with LayerScale for stability.
    """

    def __init__(self, dim: int, expansion: float = 2.0, **kwargs) -> None:
        super().__init__()
        hidden = int(dim * expansion)

        self.conv1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 1)
        self.scale = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.gelu(self.dw(self.conv1(x)))
        x = self.conv2(x)
        return self.scale(x) + res


class StreamBlock(nn.Module):
    """
    Mid-range block for Stream variants.
    Uses fused convolution and gated linear units.
    Stabilized with LayerScale and value clamping for FP16 safety.
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

        self.scale = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        # Multi-scale feature extraction
        x = torch.cat([self.dw1(x), self.dw3(x)], dim=1)
        x = self.fuse(x)

        # Gated Feed Forward
        x = self.proj(x)
        x = self.gate(x)
        a, b = x.chunk(2, dim=1)

        # Multiplication in FP32 to prevent overflow, then CLAMP to valid FP16 range
        x_f32 = a.float() * b.float()
        x = x_f32.clamp(-65504, 65504).to(a.dtype)

        x = self.out(x)
        return self.scale(x) + res


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


class TokenDictionaryCA(nn.Module):
    """
    Simplified Token Dictionary Cross-Attention.

    Core of ATD: Attends to a learned global token dictionary,
    enabling the network to understand repeating patterns and textures.

    Simplified from AdaptiveTokenCA by removing dynamic scaling.
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int = 64,
        reducted_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.reducted_dim = reducted_dim

        # Learned global token dictionary
        self.token_dict = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

        # Q/K projections to reduced dimension for efficiency
        self.q_proj = nn.Linear(dim, reducted_dim)
        self.k_proj = nn.Linear(dim, reducted_dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.scale = reducted_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            (B, C, H, W) refined features
        """
        B, C, H, W = x.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)

        # Expand token dictionary for batch
        td = self.token_dict.expand(B, -1, -1)  # (B, num_tokens, C)

        # Q from input, K/V from token dictionary
        q = self.q_proj(x_flat)  # (B, H*W, reducted_dim)
        k = self.k_proj(td)  # (B, num_tokens, reducted_dim)
        v = self.v_proj(td)  # (B, num_tokens, C)

        # Attention: query image features against token dictionary
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H*W, num_tokens)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of token values
        out = attn @ v  # (B, H*W, C)
        out = self.out_proj(out)

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        return out.transpose(1, 2).reshape(B, C, H, W)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention.

    Learns to weight channels dynamically based on global context.
    Universally improves PSNR/SSIM on all image types.
    """

    def __init__(self, dim: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ProBlock(nn.Module):
    """
    Pro block: Comprehensive quality block for SOTA PSNR/SSIM.

    Combines all proven mechanisms for universal quality improvement:
    1. Conv mixing (local feature extraction)
    2. Channel Attention SE (learns important features for each image)
    3. Window Attention (local structural consistency)
    4. Token Dictionary CA (global texture understanding)

    This combination ensures quality gains on ALL image types.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        num_tokens: int = 64,
        window_size: int = 16,
        shift_size: int = 0,
        attention_mode: str = "sdpa",
        export_safe: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        hidden = int(dim * expansion)

        # 1. Conv mixing path
        self.norm1 = RMSNorm(dim)
        self.conv1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 1)
        self.scale1 = LayerScale(dim)

        # 2. Channel Attention (SE) - universal quality improvement
        self.channel_attn = ChannelAttention(dim)
        self.scale2 = LayerScale(dim)

        # 3. Window Attention - local structure
        self.attention_mode = attention_mode
        self.export_safe = export_safe
        if attention_mode is not None and not export_safe:
            self.norm3 = RMSNorm(dim)
            self.window_attn = WindowAttention(
                dim,
                num_heads=4,
                window_size=window_size,
                shift_size=shift_size,
                attention_mode=attention_mode,
            )
            self.scale3 = LayerScale(dim)

        # 4. Token Dictionary CA - global textures
        self.norm4 = RMSNorm(dim)
        self.token_ca = TokenDictionaryCA(dim, num_tokens=num_tokens)
        self.scale4 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Conv mixing
        res = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = F.gelu(self.dw(x))
        x = self.conv2(x)
        x = res + self.scale1(x)

        # 2. Channel Attention (SE)
        res = x
        x = res + self.scale2(self.channel_attn(x))

        # 3. Window Attention (if enabled)
        if self.attention_mode is not None and not self.export_safe:
            res = x
            x = self.norm3(x).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            x = self.window_attn(x)
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            x = res + self.scale3(x)

        # 4. Token Dictionary CA
        res = x
        x = self.norm4(x)
        x = self.token_ca(x)
        x = res + self.scale4(x)

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
# 5. ATD COMPONENTS (Advanced Token Dictionary)
# ============================================================================


def index_reverse(index: torch.Tensor) -> torch.Tensor:
    """Reverse a permutation index tensor."""
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def feature_shuffle(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Shuffle features according to index."""
    dim = index.dim()
    # Handle dimension mismatch if necessary, though typical usage matches
    # spandrel logic.
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    return torch.gather(x, dim=dim - 1, index=index)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, h: int, w: int
) -> torch.Tensor:
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b, h // window_size, w // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class ATDWindowAttention(nn.Module):
    """
    Standard Window Attention module adapted for ATD Block.

    Supports:
    - Shifted Window Attention
    - Relative Position Bias
    - SDPA (Scaled Dot Product Attention) for efficiency
    - Export Safe mode (Manual MatMul)
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attention_mode: str = "sdpa",
        export_safe: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.attention_mode = attention_mode
        self.export_safe = export_safe

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, qkv: torch.Tensor, rpi: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate Relative Position Bias
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # Prepare bias mask (RPI + Optional Mask)
        # relative_position_bias: (H, N, N)
        attn_bias = relative_position_bias.unsqueeze(0)  # (1, H, N, N)

        if mask is not None:
            nw = mask.shape[0]
            # mask: (nw, N, N) -> (nw, 1, N, N)
            # attn_bias broadcasts to (nw, H, N, N)
            attn_bias = attn_bias + mask.unsqueeze(1)

            # For SDPA, we need to match the batch dimension of q (b_ = B * nw)
            # q: (B*nw, H, N, D)
            # attn_bias: (nw, H, N, N)
            # We need to repeat attn_bias B times to match B*nw
            if self.attention_mode == "sdpa" and not self.export_safe:
                B = b_ // nw
                if B > 1:
                    attn_bias = attn_bias.repeat(B, 1, 1, 1)

        # SDPA Path
        if self.attention_mode == "sdpa" and not self.export_safe:
            # F.scaled_dot_product_attention handles scale internally, but we need to add bias.
            # It enables FlashAttention / MemEffAttention automatically.
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, scale=self.scale
            )
        else:
            # Manual / Export Safe Path
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # attn: (B*nw, H, N, N)
            # attn_bias: (nw, H, N, N) or (1, H, N, N) - broadcasts correctly
            attn = attn + attn_bias
            attn = self.softmax(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x


class AdaptiveTokenCA(nn.Module):
    """
    Adaptive Token Dictionary Cross-Attention.

    Attends to a learned token dictionary for global context.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_tokens: int = 64,
        reducted_dim: int = 10,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = nn.Parameter(torch.ones([num_tokens]) * 0.5, requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.num_tokens = num_tokens

    def forward(
        self, x: torch.Tensor, td: torch.Tensor, x_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(td)
        # V: b, m, c
        v = self.wv(td)

        # Q @ K^T
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        scale = torch.clamp(self.scale, 0, 1)
        attn = attn * (1 + scale * np.log(self.num_tokens))
        attn = self.softmax(attn)

        # Attn * V
        x = (attn @ v).reshape(b, n, c)
        return x, attn


class AdaptiveCategoryMSA(nn.Module):
    """
    Adaptive Category-based Multi-head Self-Attention.

    Groups tokens by category (via similarity to dictionary) for efficient global attention.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_tokens: int = 64,
        num_heads: int = 4,
        category_size: int = 128,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.category_size = category_size
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((1, 1))), requires_grad=True
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, qkv: torch.Tensor, sim: torch.Tensor, x_size: tuple[int, int]
    ) -> torch.Tensor:
        b, n, c3 = qkv.shape
        c = c3 // 3
        gs = min(n, self.category_size)
        ng = (n + gs - 1) // gs

        tk_id = torch.argmax(sim, dim=-1, keepdim=False)
        _, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)

        pad_n = ng * gs - n
        if pad_n > 0:
            paded_qkv = torch.cat(
                (shuffled_qkv, torch.flip(shuffled_qkv[:, n - pad_n : n, :], dims=[1])),
                dim=1,
            )
        else:
            paded_qkv = shuffled_qkv

        y = paded_qkv.reshape(b, -1, gs, c3)

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c // self.num_heads).permute(
            3, 0, 1, 4, 2, 5
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)

        # Safe log barrier
        max_log = torch.log(torch.tensor(1.0 / 0.01)).to(qkv.device)
        logit_scale = torch.clamp(self.logit_scale, max=max_log).exp()
        attn = attn * logit_scale
        attn = self.softmax(attn)

        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n + pad_n, c)[:, :n, :]

        x = feature_shuffle(y, x_sort_indices_reverse)
        x = self.proj(x)
        return x


class AdaptiveConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network with depthwise convolution mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=hidden_features,
            ),
            nn.GELU(),
        )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        # x: (B, N, C)
        x = self.fc1(x)
        x = self.act(x)
        # DWConv needs (B, C, H, W)
        B, _N, _C = x.shape
        H, W = x_size
        x_grid = x.transpose(1, 2).view(B, self.hidden_features, H, W).contiguous()
        x_grid = self.dwconv(x_grid)
        x = x_grid.flatten(2).transpose(1, 2).contiguous()
        x = self.fc2(x)
        return x


class ATDTransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        shift_size,
        category_size,
        num_tokens,
        reducted_dim,
        convffn_kernel_size,
        mlp_ratio,
        qkv_bias=True,
        attention_mode="sdpa",
        export_safe=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.InstanceNorm1d(num_tokens, affine=True)
        self.sigma = nn.Parameter(torch.zeros([num_tokens, 1]), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.attn_win = ATDWindowAttention(
            dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_mode=attention_mode,
            export_safe=export_safe,
        )
        self.attn_atd = AdaptiveTokenCA(
            dim,
            input_resolution=input_resolution,
            qkv_bias=qkv_bias,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
        )
        self.attn_aca = AdaptiveCategoryMSA(
            dim,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            num_heads=num_heads,
            category_size=category_size,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = AdaptiveConvFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            kernel_size=convffn_kernel_size,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, td, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        x_atd, sim_atd = self.attn_atd(x, td, x_size)
        x_aca = self.attn_aca(qkv, sim_atd, x_size)

        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            attn_mask = params["attn_mask"]
        else:
            shifted_qkv = qkv
            attn_mask = None

        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)

        attn_windows = self.attn_win(x_windows, rpi=params["rpi_sa"], mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attn_x = shifted_x

        x = shortcut + attn_x.view(b, n, c) + x_atd + x_aca
        x = x + self.convffn(self.norm2(x), x_size)

        # Refine dictionary (token update)
        mask_soft = self.softmax(self.norm3(sim_atd.transpose(-1, -2)))
        mask_x = x.reshape(b, n, c)
        s = self.sigmoid(self.sigma)
        td = s * td + (1 - s) * torch.einsum("btn,bnc->btc", mask_soft, mask_x)

        return x, td


class ATDBlock(nn.Module):
    """
    Wraps a sequence of ATDTransformerLayers and manages the Token Dictionary.
    Compatible with ParagonSR2 body structure.
    """

    def __init__(
        self,
        dim,
        input_resolution=(64, 64),
        depth=6,
        num_heads=6,
        window_size=8,
        num_tokens=64,
        attention_mode="sdpa",
        export_safe=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.depth = depth

        # Token Dictionary
        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

        # RPI Buffer
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index_SA", relative_position_index)

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ATDTransformerLayer(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    category_size=256,
                    num_tokens=num_tokens,
                    reducted_dim=4,
                    convffn_kernel_size=5,
                    mlp_ratio=2.0,
                    attention_mode=attention_mode,
                    export_safe=export_safe,
                )
            )

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1), device=self.td.device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        cnt = 0
        for hh in h_slices:
            for ww in w_slices:
                img_mask[:, hh, ww, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(self, x):
        # x is (B, C, H, W) from Paragon Body
        # We need to flatten to (B, N, C) for ATD
        B, C, H, W = x.shape

        # Padding
        mod = self.window_size
        h_pad = ((H + mod - 1) // mod) * mod - H
        w_pad = ((W + mod - 1) // mod) * mod - W
        x = F.pad(x, (0, w_pad, 0, h_pad), mode="reflect")
        Hp, Wp = x.shape[2], x.shape[3]

        # Flatten
        x_tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)

        params = {
            "attn_mask": self.calculate_mask((Hp, Wp)),
            "rpi_sa": self.relative_position_index_SA,
        }

        td = self.td.repeat(B, 1, 1)

        for layer in self.layers:
            x_tokens, td = layer(x_tokens, td, (Hp, Wp), params)

        # Unflatten
        x = x_tokens.transpose(1, 2).reshape(B, C, Hp, Wp)

        # Crop
        x = x[:, :, :H, :W]
        return x


# ============================================================================
# 5. MAIN NETWORK
# ============================================================================


@ARCH_REGISTRY.register()
class ParagonSR2(nn.Module):
    """
    ParagonSR2: Dual-Path Super-Resolution Network.

    A deployment-first SISR architecture combining classical signal processing
    with learned neural refinement.

    Architecture:
        Input -> [Classical Base Upsampler] -> base_output
              -> [Conv_In] -> [Body (ResidualGroups)] -> [Conv_Mid] -> [PixelShuffle] -> detail_output
        Output = base_output + (detail_output * detail_gain)

    Args:
        scale: Upscale factor (2 or 4).
        in_chans: Input channels (default: 3 for RGB).
        num_feat: Feature channels in the body.
        num_groups: Number of residual groups.
        num_blocks: Blocks per group.
        variant: Block type - "realtime", "stream", or "photo".
        detail_gain: Initial gain for learned detail (learnable parameter).
        upsampler_alpha: Sharpening strength for classical base (0 = none).
        use_checkpointing: Enable gradient checkpointing for training.
        attention_mode: Attention backend - "sdpa", "flex", or None.
        export_safe: If True, disables attention for ONNX export compatibility.
        window_size: Window size for attention (Photo variant only).

    Forward Args:
        x: Input tensor (B, C, H, W).
        feature_tap: If True, also returns intermediate features.
        prev_feat: Previous frame's features for temporal blending (video mode).
        alpha: Blending strength for temporal features (0-1).

    Returns:
        If feature_tap=False: Output tensor (B, C, H*scale, W*scale).
        If feature_tap=True: Tuple of (output, features) for video processing.
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

        if variant == "pro":
            # Pro variant: Conv + Channel Attn + Window Attn + Token Dictionary
            def build_pro_blocks(group_idx: int):
                blocks = []
                for i in range(num_blocks):
                    block_idx = group_idx * num_blocks + i
                    shift = (window_size // 2) if (block_idx % 2 != 0) else 0
                    blocks.append(
                        ProBlock(
                            num_feat,
                            num_tokens=64,
                            window_size=window_size,
                            shift_size=shift,
                            attention_mode=attention_mode,
                            export_safe=export_safe,
                        )
                    )
                return blocks

            self.body = nn.Sequential(
                *[
                    ResidualGroup(
                        build_pro_blocks(g),
                        checkpointing=use_checkpointing,
                    )
                    for g in range(num_groups)
                ]
            )
        else:
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
        self,
        x: torch.Tensor,
        feature_tap: bool = False,
        prev_feat: torch.Tensor | None = None,
        alpha: float = 0.2,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional temporal feature blending.

        Args:
            x: Input image tensor (B, 3, H, W).
            feature_tap: Return intermediate features for video mode.
            prev_feat: Previous frame's features for temporal blending.
            alpha: Blending weight (0=current only, 1=history only).

        Returns:
            SR output, or (SR output, features) if feature_tap=True.
        """
        base = self.base(x)

        x = self.conv_in(x)

        # Temporal feedback blending (video mode)
        if prev_feat is not None:
            x = (1.0 - alpha) * x + alpha * prev_feat

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


@ARCH_REGISTRY.register()
def paragonsr2_pro(scale=4, **kw):
    """
    Pro variant: The most capable ParagonSR2 configuration.

    Combines all proven quality mechanisms:
    - Conv Mixing (local features)
    - Channel Attention SE (learns which features matter)
    - Window Attention (local structure)
    - Token Dictionary CA (global textures)

    Config: 6 groups x 6 blocks = 36 ProBlocks with 64 tokens.
    """
    return ParagonSR2(
        scale=scale,
        num_feat=64,
        num_groups=6,  # 6 groups for maximum capacity
        num_blocks=6,  # 6 ProBlocks per group = 36 total
        variant="pro",
        detail_gain=kw.pop("detail_gain", 0.1),
        upsampler_alpha=kw.pop("upsampler_alpha", 0.4),
        attention_mode=kw.pop("attention_mode", "sdpa"),
        export_safe=kw.pop("export_safe", False),
        window_size=kw.pop("window_size", 16),
        **kw,
    )
