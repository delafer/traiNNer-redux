import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY

from .resampler import MagicKernelSharp2021Upsample


# -------------------------
# Reuse / minor refactors
# -------------------------
class DownBlock(nn.Sequential):
    """Downsampling block for the MUNet discriminator (same as before)."""

    def __init__(self, in_feat: int, out_feat: int, slope: float = 0.2) -> None:
        super().__init__(
            spectral_norm(nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )


class UpBlock(nn.Module):
    """
    Upsampling block for the MUNet discriminator (keeps your Magic upsampler).
    Produces features aligned to a skip connection and then fuses them.
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

        self.magic_upsample = MagicKernelSharp2021Upsample(in_feat)
        self.post_upsample_conv = spectral_norm(
            nn.Conv2d(in_feat, skip_feat, 3, 1, 1, bias=False)
        )

        fusion_in_channels = skip_feat + skip_feat
        self.fusion_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(fusion_in_channels, out_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # scale ratios
        scale_h = skip.shape[2] / x.shape[2]
        scale_w = skip.shape[3] / x.shape[3]

        if abs(scale_h - 1.0) < 1e-6 and abs(scale_w - 1.0) < 1e-6:
            x = self.magic_upsample(x, 1.0)
        else:
            x = self.magic_upsample(x, (scale_h, scale_w))

        x = self.post_upsample_conv(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        return self.fusion_conv(x)


# -------------------------
# Frequency helpers
# -------------------------
def luminance_weights_conv():
    """1x1 conv that converts RGB -> luminance using standard weights (not learnable)."""
    conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        # Rec. 601 luma coefficients
        w = torch.tensor([[[[0.2989]], [[0.5870]], [[0.1140]]]])  # shape (1,3,1,1)
        conv.weight.copy_(w)
    for p in conv.parameters():
        p.requires_grad = False
    return conv


# -------------------------
# New improved multi-branch frequency-aware MUNet
# -------------------------
@ARCH_REGISTRY.register()
class MUNet(nn.Module):
    """
    Multi-branch, frequency-aware Magic U-Net discriminator.

    Branches:
      - spatial (U-Net decoder) : global + local spatial reasoning (your original path)
      - freq (FFT magnitude processed) : explicit frequency-domain detection
      - patch (texture branch from bottleneck) : texture/patch-level cues

    The branches share an encoder backbone to keep parameter efficiency and
    produce complementary gradients for the generator.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_feat: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.slope = slope
        self.num_feat = int(num_feat)

        # ---- shared shallow input conv ----
        self.in_conv = spectral_norm(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))

        # ---- encoder (shared) ----
        self.down_blocks = nn.ModuleList()
        encoder_channels = [num_feat]
        in_ch = num_feat
        for mult in ch_mult:
            out_ch = num_feat * mult
            self.down_blocks.append(DownBlock(in_ch, out_ch, slope))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        # ---- bottleneck ----
        self.mid_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # ---- spatial decoder (U-Net style) ----
        self.up_blocks = nn.ModuleList()
        decoder_specs = list(reversed(encoder_channels[:-1]))
        in_ch = encoder_channels[-1]
        for skip_ch in decoder_specs:
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_feat=skip_ch, slope=slope)
            )
            in_ch = skip_ch

        # ---- frequency branch (explicit FFT magnitude stream) ----
        # luminance conv (fixed) + small conv stem to extract freq features
        self.freq_luma = luminance_weights_conv()
        # small conv stack to process log-magnitude FFT map -> produce num_feat channels
        self.freq_proc = nn.Sequential(
            spectral_norm(nn.Conv2d(1, num_feat // 2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(num_feat // 2, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # ---- patch/texture branch (from bottleneck) ----
        # reduce channels then upsample to input resolution and refine
        self.patch_reduce = nn.Sequential(
            spectral_norm(
                nn.Conv2d(encoder_channels[-1], num_feat, 1, 1, 0, bias=False)
            ),
            nn.LeakyReLU(slope, inplace=True),
        )
        self.patch_upsample = nn.Sequential(
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # ---- fusion head ----
        # will concat [spatial_features, freq_features, patch_features] -> reduce to num_feat -> out
        fusion_in = num_feat * 3
        self.fusion_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(fusion_in, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
            spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)),
            nn.LeakyReLU(slope, inplace=True),
        )

        # ---- final output conv -> single-channel map (like original) ----
        self.out_conv = spectral_norm(nn.Conv2d(num_feat, 1, 3, 1, 1))

        # weight init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
            )

    # -------------------------
    # Frequency utilities
    # -------------------------
    @staticmethod
    def _fft_log_magnitude(x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Compute log-magnitude of 2D FFT of a luminance map (B,1,H,W) -> (B,1,H,W).
        This is differentiable via torch.fft.
        """
        # expect x shape (B,1,H,W)
        # remove channel dim for FFT, perform per-sample 2D FFT
        b, c, h, w = x.shape
        assert c == 1, "fft input must have single channel (luma)"
        # squeeze channel
        x2 = x.view(b, h, w)
        # compute complex FFT
        fft = torch.fft.fft2(x2, norm="ortho")  # (B,H,W) complex
        mag = torch.abs(fft)  # (B,H,W) real
        # log scaling - stabilise
        log_mag = torch.log(mag + eps)
        # return as (B,1,H,W)
        return log_mag.unsqueeze(1)

    # -------------------------
    # forward helpers
    # -------------------------
    def _run_shared_encoder(self, x: Tensor):
        x = self.in_conv(x)
        skips = [x]
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        return x, skips

    def _run_spatial_decoder(self, bottleneck: Tensor, skips: list[Tensor]) -> Tensor:
        x = bottleneck
        # we pop last skip in original implementation; preserve same behavior
        skips = skips.copy()
        skips.pop()
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip)
        return x

    def _run_frequency_branch(self, orig: Tensor) -> Tensor:
        # orig: (B,3,H,W)
        # convert to luma
        with torch.no_grad():
            luma = self.freq_luma(
                orig
            )  # (B,1,H,W)  - fixed conv, non-learnable weights
        # compute log-magnitude FFT (differentiable)
        log_mag = self._fft_log_magnitude(luma)  # (B,1,H,W)
        # process with small conv stack -> (B,num_feat,H,W)
        freq_feat = self.freq_proc(log_mag)
        return freq_feat

    def _run_patch_branch(
        self, bottleneck: Tensor, target_hw: tuple[int, int]
    ) -> Tensor:
        # reduce channels
        x = self.patch_reduce(bottleneck)  # (B,num_feat, h_b, w_b)
        # upsample to target resolution (input resolution)
        # Use nearest upsample multiple times to match spatial dims (simple, robust)
        x = F.interpolate(x, size=target_hw, mode="nearest")
        # refine
        x = self.patch_upsample(x)
        return x

    # -------------------------
    # main forward
    # -------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward returns a prediction map (B,1,H,W) similar to original MUNet.
        """
        # Shared encoder
        bottleneck, skips = self._run_shared_encoder(x)

        # Bottleneck conv
        bottleneck = self.mid_conv(bottleneck)

        # Spatial decoder (U-Net path)
        spatial_feat = self._run_spatial_decoder(
            bottleneck, skips
        )  # (B,num_feat,H,W) expected

        # Frequency branch (operates on original image)
        freq_feat = self._run_frequency_branch(x)  # (B,num_feat,H,W) after freq_proc

        # Patch/texture branch (from bottleneck, upsampled to input resol)
        target_hw = (spatial_feat.shape[2], spatial_feat.shape[3])
        patch_feat = self._run_patch_branch(bottleneck, target_hw)  # (B,num_feat,H,W)

        # Ensure all feature maps have same spatial dims
        if not (spatial_feat.shape[2:] == freq_feat.shape[2:] == patch_feat.shape[2:]):
            # Resize any mismatch to spatial_feat dims
            fh, fw = spatial_feat.shape[2], spatial_feat.shape[3]
            freq_feat = F.interpolate(
                freq_feat, size=(fh, fw), mode="bilinear", align_corners=False
            )
            patch_feat = F.interpolate(
                patch_feat, size=(fh, fw), mode="bilinear", align_corners=False
            )

        # Concatenate along channels
        fused = torch.cat(
            [spatial_feat, freq_feat, patch_feat], dim=1
        )  # (B, 3*num_feat, H, W)
        fused = self.fusion_conv(fused)
        out = self.out_conv(fused)
        return out

    def forward_with_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Forward that returns both prediction and intermediate features.

        Returns:
            pred: final discriminator output (B,1,H,W)
            feats: list of multi-scale, multi-branch intermediate activations
        """
        # Shared encoder
        bottleneck, skips = self._run_shared_encoder(x)

        # Collect features at different scales
        feats = []

        # Add encoder features (at different scales)
        for skip in skips:
            feats.append(skip)

        # Bottleneck conv
        bottleneck = self.mid_conv(bottleneck)
        feats.append(bottleneck)

        # Spatial decoder (U-Net path)
        spatial_feat = self._run_spatial_decoder(
            bottleneck, skips.copy()
        )  # (B,num_feat,H,W) expected

        # Frequency branch (operates on original image)
        freq_feat = self._run_frequency_branch(x)  # (B,num_feat,H,W) after freq_proc

        # Patch/texture branch (from bottleneck, upsampled to input resol)
        target_hw = (spatial_feat.shape[2], spatial_feat.shape[3])
        patch_feat = self._run_patch_branch(bottleneck, target_hw)  # (B,num_feat,H,W)

        # Ensure all feature maps have same spatial dims
        if not (spatial_feat.shape[2:] == freq_feat.shape[2:] == patch_feat.shape[2:]):
            # Resize any mismatch to spatial_feat dims
            fh, fw = spatial_feat.shape[2], spatial_feat.shape[3]

            freq_feat = F.interpolate(
                freq_feat, size=(fh, fw), mode="bilinear", align_corners=False
            )
            patch_feat = F.interpolate(
                patch_feat, size=(fh, fw), mode="bilinear", align_corners=False
            )

        # Add branch features before fusion
        feats.extend([spatial_feat, freq_feat, patch_feat])

        # Concatenate along channels
        fused = torch.cat(
            [spatial_feat, freq_feat, patch_feat], dim=1
        )  # (B, 3*num_feat, H, W)
        fused = self.fusion_conv(fused)

        # Add fused feature
        feats.append(fused)

        out = self.out_conv(fused)
        return out, feats


def forward_with_features(self, x):
    """
    Returns:
        pred: final discriminator output
        feats: list of multi-scale, multi-branch intermediate activations
    """
    feats = []
    out, branch_feats = self.forward_return_features(x)

    # Merge multi-branch features
    for bf in branch_feats:
        feats.extend(bf)

    return out, feats
