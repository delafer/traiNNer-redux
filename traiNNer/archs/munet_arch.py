import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY

from .resampler import MagicKernelSharp2021Upsample


class DownBlock(nn.Sequential):
    """Downsampling block for the MUNet discriminator."""

    def __init__(self, in_feat: int, out_feat: int, slope: float = 0.2) -> None:
        super().__init__(
            spectral_norm(nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )


class UpBlock(nn.Module):
    """
    Upsampling block for the MUNet discriminator.

    Replaces bilinear interpolation with MagicKernelSharp2021Upsample
    to preserve anti-aliasing and sharpness characteristics.
    """

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

        # Upsampling layer with the custom Magic kernel resampler
        self.magic_upsample = MagicKernelSharp2021Upsample(in_feat)

        # Convolution to reduce aliasing and align channel count with the skip connection
        self.post_upsample_conv = spectral_norm(
            nn.Conv2d(in_feat, skip_feat, 3, 1, 1, bias=False)
        )

        fusion_in_channels = skip_feat + skip_feat

        # Fusion layer (processes concatenated skip connection)
        self.fusion_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(fusion_in_channels, out_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # Compute scale ratios relative to the skip connection
        scale_h = skip.shape[2] / x.shape[2]
        scale_w = skip.shape[3] / x.shape[3]

        if abs(scale_h - 1.0) < 1e-6 and abs(scale_w - 1.0) < 1e-6:
            x = self.magic_upsample(x, 1.0)
        else:
            x = self.magic_upsample(x, (scale_h, scale_w))

        x = self.post_upsample_conv(x)

        # Align feature map shapes exactly
        if x.shape[2:] != skip.shape[2:]:
            # Tiny resize correction
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        return self.fusion_conv(x)


@ARCH_REGISTRY.register()
class MUNet(nn.Module):
    """
    Magic U-Net (MUNet): A U-Net discriminator using MagicKernelSharp2021Upsample.

    Advantages over standard U-Net discriminators:
      - Anti-aliased upsampling with pre-sharpened magic kernel
      - Consistent spectral normalization for discriminator stability
      - Grad-friendly, numerically stable in mixed-precision
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        encoder_channels = [num_feat]
        in_ch = num_feat
        for mult in ch_mult:
            out_ch = num_feat * mult
            self.down_blocks.append(DownBlock(in_ch, out_ch, slope))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        # --- Bottleneck ---
        self.mid_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        decoder_specs = list(reversed(encoder_channels[:-1]))
        in_ch = encoder_channels[-1]
        for skip_ch in decoder_specs:
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_feat=skip_ch, slope=slope)
            )
            in_ch = skip_ch

        # --- Output ---
        self.out_conv = spectral_norm(nn.Conv2d(num_feat, 1, 3, 1, 1))

        # Optional weight init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
            )

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        x = self.in_conv(x)
        skips = [x]

        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        # Bottleneck
        x = self.mid_conv(x)

        # Decoder
        skips.pop()  # drop last skip (same resolution as bottleneck)
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip)

        # Output prediction map
        return self.out_conv(x)
