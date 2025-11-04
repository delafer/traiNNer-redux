import math

import torch
import torch.nn.functional as F
from torch import nn

# This implementation is inspired by the discussion with the user and a deeper
# reading of the Magic Kernel author's methodology. The key insight is that
# for upsampling, the sharpening should happen *before* the main resampling kernel is applied.


def get_magic_kernel_weights() -> torch.Tensor:
    """
    This is the main, smooth, 3rd-order Magic B-spline kernel.
    It's known for excellent anti-aliasing properties.
    """
    return torch.tensor([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])


def get_magic_sharp_2021_kernel_weights() -> torch.Tensor:
    """
    This is the 7-tap sharpening kernel. It's a corrective filter.
    """
    return torch.tensor([-1 / 32, 0, 9 / 32, 16 / 32, 9 / 32, 0, -1 / 32])


class SeparableConv(nn.Module):
    """
    A helper module to perform a 1D separable convolution on a 2D image.
    """

    def __init__(self, in_channels: int, kernel: torch.Tensor) -> None:
        super().__init__()
        kernel_size = len(kernel)

        # Horizontal convolution
        self.conv_h = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=in_channels,
            bias=False,
        )
        self.conv_h.weight.data = kernel.view(1, 1, 1, -1).repeat(in_channels, 1, 1, 1)
        self.conv_h.weight.requires_grad = False

        # Vertical convolution
        self.conv_v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=in_channels,
            bias=False,
        )
        self.conv_v.weight.data = kernel.view(1, 1, -1, 1).repeat(in_channels, 1, 1, 1)
        self.conv_v.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_v(self.conv_h(x))


class MagicKernelSharp2021Upsample(nn.Module):
    """
    A more faithful implementation of Magic Kernel Sharp 2021 for upsampling.
    It correctly applies a pre-sharpening filter before resampling with the
    main smooth kernel.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # Stage 1: The pre-sharpening filter
        sharp_kernel = get_magic_sharp_2021_kernel_weights()
        self.sharpen = SeparableConv(in_channels, sharp_kernel)

        # Stage 2: The main resampling filter
        # For resampling, we can use a simpler convolution pass on upscaled data.
        # This is a common and efficient technique in PyTorch for custom kernels.
        resample_kernel = get_magic_kernel_weights()
        self.resample_conv = SeparableConv(in_channels, resample_kernel)

    def forward(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        # Step 1: Apply the 7-tap sharpening filter to the low-resolution input.
        x_sharpened = self.sharpen(x)

        # Step 2: Upsample using simple nearest neighbor. This is fast and avoids
        # introducing any new artifacts before our high-quality filter is applied.
        x_upsampled = F.interpolate(
            x_sharpened, scale_factor=scale_factor, mode="nearest"
        )

        # Step 3: Apply the main smooth Magic Kernel to the upsampled data.
        # This acts as a high-quality anti-aliasing and reconstruction filter,
        # smoothing the blockiness of the nearest-neighbor resize.
        return self.resample_conv(x_upsampled)
