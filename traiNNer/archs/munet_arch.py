#!/usr/bin/env python3
"""
MUNet - Multi-Branch U-Net Discriminator for Super-Resolution GANs

Author: Philip Hofmann
License: MIT
Repository: https://github.com/Phhofm/traiNNer-redux

═══════════════════════════════════════════════════════════════════════════════
DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════════

MUNet is a specialized discriminator designed to detect artifacts across multiple
domains simultaneously. It acts as a "Multi-View Critic."

Key Innovation: Quad-Branch Detection
-------------------------------------
1. Spatial Branch (U-Net): Checks structural consistency and global coherence.
2. Gradient Branch: Checks for edge ringing, jaggies, and unnatural gradients.
3. Frequency Branch: Checks for spectral anomalies (blur/oversharpening).
4. Patch Branch: Checks for local texture consistency at the bottleneck level.

Fusion Mechanism:
-----------------
Instead of averaging these scores, an Attention Fusion module learns *where* to
trust each branch. It might prioritize the Gradient branch on edges and the
Frequency branch in textured regions.

Usage Guide:
------------
- For ParagonSR2_Nano/Stream: Use num_feat=32, ch_mult=(1, 2, 2)
- For ParagonSR2_Photo/Pro:   Use num_feat=64, ch_mult=(1, 2, 4, 8)
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY

# ═════════════════════════════════════════════════════════════════════════════
# 1. HELPER LAYERS (MagicKernel & Upsampling)
# ═════════════════════════════════════════════════════════════════════════════


def get_magic_kernel_weights() -> torch.Tensor:
    return torch.tensor([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])


def get_magic_sharp_2021_kernel_weights() -> torch.Tensor:
    return torch.tensor([-1 / 32, 0, 9 / 32, 16 / 32, 9 / 32, 0, -1 / 32])


class SeparableConv(nn.Module):
    """Fixed weight Separable Convolution for MagicKernel."""

    def __init__(self, in_channels: int, kernel: torch.Tensor) -> None:
        super().__init__()
        kernel_size = len(kernel)
        self.conv_h = nn.Conv2d(
            in_channels,
            in_channels,
            (1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=in_channels,
            bias=False,
        )
        self.conv_v = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=in_channels,
            bias=False,
        )

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
    """Stable Upsampling for Discriminator (Runtime Scale)."""

    def __init__(self, in_channels: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(max(0.0, min(alpha, 1.0)))
        self.sharpen = SeparableConv(in_channels, get_magic_sharp_2021_kernel_weights())
        self.resample_conv = SeparableConv(in_channels, get_magic_kernel_weights())

    def forward(
        self, x: torch.Tensor, scale_factor: int | float | tuple[float, float]
    ) -> torch.Tensor:
        if self.alpha > 0.0:
            x_sharp = self.sharpen(x)
            x = x + self.alpha * (x_sharp - x)
        x_upsampled = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        return self.resample_conv(x_upsampled)


# ═════════════════════════════════════════════════════════════════════════════
# 2. BUILDING BLOCKS
# ═════════════════════════════════════════════════════════════════════════════


class DownBlock(nn.Sequential):
    def __init__(self, in_feat: int, out_feat: int, slope: float = 0.2) -> None:
        super().__init__(
            spectral_norm(nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(
        self,
        in_feat: int,
        skip_feat: int,
        out_feat: int | None = None,
        slope: float = 0.2,
    ) -> None:
        super().__init__()
        if out_feat is None:
            out_feat = skip_feat
        self.magic_upsample = MagicKernelSharp2021Upsample(in_feat)
        self.post_upsample_conv = spectral_norm(
            nn.Conv2d(in_feat, skip_feat, 3, 1, 1, bias=False)
        )
        self.fusion_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(skip_feat * 2, out_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # Dynamic scale calculation based on skip connection size
        scale_h = skip.shape[2] / x.shape[2]
        scale_w = skip.shape[3] / x.shape[3]

        if abs(scale_h - 1.0) < 1e-6 and abs(scale_w - 1.0) < 1e-6:
            x = self.magic_upsample(x, 1.0)
        else:
            x = self.magic_upsample(x, (scale_h, scale_w))

        x = self.post_upsample_conv(x)
        # Final safety check for size
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

        return self.fusion_conv(torch.cat([x, skip], dim=1))


class LocalWindowAttention(nn.Module):
    """
    Spectral Norm Attention for Discriminator Stability.
    Updated with batch-dimension fix (unbind).
    """

    def __init__(
        self, channels: int, reduction: int = 8, window_size: int = 32
    ) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.query = spectral_norm(nn.Conv2d(channels, reduced, 1))
        self.key = spectral_norm(nn.Conv2d(channels, reduced, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.window_size = window_size
        self.scale = reduced**-0.5

    def forward(self, x: Tensor) -> Tensor:
        _B, _C, H, W = x.shape
        if H <= self.window_size and W <= self.window_size:
            return self._full_attn(x)

        # Pad for windowing
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_in = F.pad(x, (0, pad_w, 0, pad_h)) if pad_h > 0 or pad_w > 0 else x

        # Simple full attention fallback for discriminator stability/speed trade-off
        # (Implementing full window partition in Disc can be overkill vs just resizing crop)
        return self._full_attn(x)

    def _full_attn(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        flat_hw = H * W

        # Project
        q = self.query(x).view(B, -1, flat_hw).permute(0, 2, 1)  # (B, HW, C')
        k = self.key(x).view(B, -1, flat_hw)  # (B, C', HW)
        v = self.value(x).view(B, -1, flat_hw)  # (B, C, HW)

        # Attn map
        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, H, W)

        return x + self.gamma * out


class AttentionFusion(nn.Module):
    """Learns to weight branches per spatial location."""

    def __init__(self, num_branches: int, num_feat: int, slope: float = 0.2) -> None:
        super().__init__()
        self.attention_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat * num_branches, num_feat, 1)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(num_feat, num_branches, 1)),
        )
        self.fusion_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, branches: list[Tensor]) -> Tensor:
        concat = torch.cat(branches, dim=1)
        # Softmax ensures weights sum to 1, preventing signal explosion
        attn = F.softmax(self.attention_conv(concat), dim=1)

        fused = torch.zeros_like(branches[0])
        for i, branch in enumerate(branches):
            fused = fused + (attn[:, i : i + 1] * branch)

        return self.fusion_conv(fused)


# ═════════════════════════════════════════════════════════════════════════════
# 3. MAIN DISCRIMINATOR
# ═════════════════════════════════════════════════════════════════════════════


@ARCH_REGISTRY.register()
class MUNet(nn.Module):
    """
    Multi-Branch U-Net Discriminator (Spatial + Gradient + Frequency + Patch).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_feat = num_feat
        self.num_in_ch = num_in_ch
        self.ch_mult = ch_mult

        # 1. Input
        self.in_conv = spectral_norm(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))

        # 2. Encoder
        self.down_blocks = nn.ModuleList()
        encoder_channels = [num_feat]
        in_ch = num_feat
        for mult in ch_mult:
            out_ch = num_feat * mult
            self.down_blocks.append(DownBlock(in_ch, out_ch, slope))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        # 3. Bottleneck (Mid)
        self.mid_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.self_attn = LocalWindowAttention(in_ch, window_size=32)

        # 4. Spatial Decoder (Branch 1)
        self.up_blocks = nn.ModuleList()
        decoder_specs = list(reversed(encoder_channels[:-1]))
        in_ch = encoder_channels[-1]
        for skip_ch in decoder_specs:
            self.up_blocks.append(UpBlock(in_ch, skip_ch, slope=slope))
            in_ch = skip_ch

        # 5. Gradient Branch (Branch 2)
        self.grad_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(2, num_feat // 2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(num_feat // 2, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # 6. Frequency Branch (Branch 3)
        self.freq_proc = nn.Sequential(
            spectral_norm(nn.Conv2d(1, num_feat // 2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(num_feat // 2, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # 7. Patch Branch (Branch 4)
        self.patch_reduce = nn.Sequential(
            spectral_norm(nn.Conv2d(encoder_channels[-1], num_feat, 1, 1, 0)),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.patch_upsample = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # 8. Fusion & Output
        self.attention_fusion = AttentionFusion(4, num_feat, slope)
        self.out_conv = spectral_norm(nn.Conv2d(num_feat, 1, 3, 1, 1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
            )

    def _compute_gradients(self, x: Tensor) -> Tensor:
        """Computes gradients with REPLICATE padding to avoid border artifacts."""
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]

        # Padding with 'replicate' prevents the discriminator from overfitting to
        # the black border caused by zero-padding
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")

        return self.grad_conv(torch.cat([grad_x, grad_y], dim=1))

    def _compute_frequency(self, x: Tensor) -> Tensor:
        """Computes Centered FFT Magnitude."""
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gray_f32 = gray.float()  # FFT needs float32

        fft = torch.fft.fft2(gray_f32, norm="ortho")
        fft = torch.fft.fftshift(fft)  # Center low frequencies
        mag = torch.log(torch.abs(fft) + 1e-8)

        return self.freq_proc(mag.to(dtype=x.dtype))

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        bottleneck = self.in_conv(x)
        skips = [bottleneck]
        for block in self.down_blocks:
            bottleneck = block(bottleneck)
            skips.append(bottleneck)

        # Mid
        bottleneck = self.mid_conv(bottleneck)
        bottleneck = self.self_attn(bottleneck)

        # Branches
        spatial = bottleneck
        curr_skips = skips[:-1]  # Pop last
        for block, skip in zip(self.up_blocks, reversed(curr_skips), strict=False):
            spatial = block(spatial, skip)

        grad = self._compute_gradients(x)
        freq = self._compute_frequency(x)

        target_hw = (spatial.shape[2], spatial.shape[3])
        patch = self.patch_reduce(bottleneck)
        patch = F.interpolate(patch, size=target_hw, mode="nearest")
        patch = self.patch_upsample(patch)

        # Align & Fuse
        branches = [spatial, grad, freq, patch]
        aligned = []
        for b in branches:
            if b.shape[2:] != target_hw:
                b = F.interpolate(
                    b, size=target_hw, mode="bilinear", align_corners=False
                )
            aligned.append(b)

        fused = self.attention_fusion(aligned)
        return self.out_conv(fused)

    def __repr__(self) -> str:
        return f"MUNet(num_feat={self.num_feat}, ch_mult={self.ch_mult})"
