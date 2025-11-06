from collections.abc import Sequence

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

        # Vertical convolution
        self.conv_v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
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
    """
    A more faithful implementation of Magic Kernel Sharp 2021 for upsampling.
    It correctly applies a pre-sharpening filter before resampling with the
    main smooth kernel.
    """

    @staticmethod
    def _resolve_target_size(
        scale_factor: int | float | Sequence[int | float],
        height: int,
        width: int,
    ) -> tuple[int, int]:
        """
        Normalize the incoming scale factor into an integer output size.

        Args:
            scale_factor: Single isotropic scale or (h, w) pair.
            height: Input height.
            width: Input width.

        Returns:
            Tuple of (target_height, target_width).
        """
        if isinstance(scale_factor, (int, float)):
            scale_h = scale_w = float(scale_factor)
        elif isinstance(scale_factor, Sequence) and len(scale_factor) == 2:
            scale_h, scale_w = (float(s) for s in scale_factor)
        else:
            raise TypeError(
                "scale_factor must be an int/float or a 2-element sequence of ints/floats."
            )

        if scale_h <= 0 or scale_w <= 0:
            raise ValueError("scale_factor values must be positive.")

        target_h = max(round(height * scale_h), 1)
        target_w = max(round(width * scale_w), 1)
        return target_h, target_w

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

    def forward(
        self, x: torch.Tensor, scale_factor: int | float | Sequence[int | float]
    ) -> torch.Tensor:
        """
        Upsample the incoming tensor using the Magic Kernel Sharp 2021 recipe.

        Args:
            x: Input tensor of shape (N, C, H, W).
            scale_factor: Isotropic scale (int/float) or anisotropic pair (h_scale, w_scale).

        Returns:
            The upsampled tensor.
        """
        # Step 1: Apply the 7-tap sharpening filter to the low-resolution input.
        x_sharpened = self.sharpen(x)

        # Step 2: Decide whether a resize is required and perform a nearest-neighbour upsample if needed.
        _, _, height, width = x_sharpened.shape
        target_h, target_w = self._resolve_target_size(scale_factor, height, width)

        if target_h != height or target_w != width:
            x_upsampled = F.interpolate(
                x_sharpened,
                size=(target_h, target_w),
                mode="nearest",
            )
        else:
            x_upsampled = x_sharpened

        # Step 3: Apply the main smooth Magic Kernel to the (optionally) upsampled data.
        # This acts as a high-quality anti-aliasing and reconstruction filter,
        # smoothing the blockiness of the nearest-neighbor resize.
        return self.resample_conv(x_upsampled)
