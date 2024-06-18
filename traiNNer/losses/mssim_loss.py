import math
from collections.abc import Sequence

import torch
from torch import SymInt, nn
from torch.nn import functional as F  # noqa: N812
from traiNNer.utils.registry import LOSS_REGISTRY


####################################
# Modified MSSIM Loss with cosine similarity from neosr
# https://github.com/muslll/neosr/blob/master/neosr/losses/ssim_loss.py
####################################


class GaussianFilter2D(nn.Module):
    def __init__(
            self,
            window_size: int = 11,
            in_channels: int = 3,
            sigma: float = 1.5,
            padding: int | SymInt | Sequence[int | SymInt] = None,
    ) -> None:
        """2D Gaussian Filer

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            padding (int, optional): The padding of the gaussian filter. Defaults to None.
                If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
        """
        super().__init__()
        self.window_size = window_size
        if not (window_size % 2 == 1):
            raise ValueError("Window size must be odd.")
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma

        kernel = self._get_gaussian_window1d()
        kernel = self._get_gaussian_window2d(kernel)
        self.register_buffer(
            name="gaussian_window", tensor=kernel.repeat(in_channels, 1, 1, 1)
        )

    def _get_gaussian_window1d(self) -> torch.Tensor:
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x ** 2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, 1, self.window_size)

    def _get_gaussian_window2d(self, gaussian_window_1d: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(
            gaussian_window_1d.transpose(dim0=-1, dim1=-2), gaussian_window_1d
        )
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(
            input=x,
            weight=self.gaussian_window,
            stride=1,
            padding=self.padding,
            groups=x.shape[1],
        )
        return x


# def dynamic_lambda_smooth(msssim, cosim, default_lambda=5, k=10):
#
#     top_threshold = 0.9
#     bottom_threshold = 0.5
#
#     dynamic_lambda = msssim / cosim
#     lambda_value = torch.zeros_like(msssim)
#
#     lower_middle = (msssim >= bottom_threshold) & (
#                 msssim <= (bottom_threshold + (top_threshold - bottom_threshold) / 2))
#     upper_middle = (msssim > (bottom_threshold + (top_threshold - bottom_threshold) / 2)) & (msssim <= top_threshold)
#
#     if torch.any(lower_middle):
#         sigmoid_factor_lower = torch.sigmoid(
#             k * ((msssim[lower_middle] - bottom_threshold) / ((top_threshold - bottom_threshold) / 2)))
#         lambda_value[lower_middle] = sigmoid_factor_lower * dynamic_lambda[lower_middle]
#
#     if torch.any(upper_middle):
#         sigmoid_factor_upper = torch.sigmoid(k * (
#                     1 - (msssim[upper_middle] - (bottom_threshold + (top_threshold - bottom_threshold) / 2)) / (
#                         (top_threshold - bottom_threshold) / 2)))
#         lambda_value[upper_middle] = sigmoid_factor_upper * default_lambda + (1 - sigmoid_factor_upper) * \
#                                      dynamic_lambda[upper_middle]
#
#     top_range = (msssim > top_threshold)
#     lambda_value[top_range] = default_lambda
#
#     return lambda_value


@LOSS_REGISTRY.register()
class MSSIMLoss(nn.Module):
    def __init__(
            self,
            window_size: int = 11,
            in_channels: int = 3,
            sigma: float = 1.5,
            k1: float = 0.01,
            k2: float = 0.03,
            l: int = 1,
            padding: int | SymInt | Sequence[int | SymInt] = None,
            loss_weight: float = 1.0,
    ) -> None:
        """Adapted from 'A better pytorch-based implementation for the mean structural
            similarity. Differentiable simpler SSIM and MS-SSIM.':
                https://github.com/lartpang/mssim.pytorch

            Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            window_size (int): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float): The sigma of the gaussian filter. Defaults to 1.5.
            k1 (float): k1 of MSSIM. Defaults to 0.01.
            k2 (float): k2 of MSSIM. Defaults to 0.03.
            L (int): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None,
                the filter will use window_size//2 as the padding. Another common setting is 0.
            loss_weight (float): Weight of final loss value.
        """
        super().__init__()

        self.window_size = window_size
        self.C1 = (k1 * l) ** 2  # equ 7 in ref1
        self.C2 = (k2 * l) ** 2  # equ 7 in ref1
        self.loss_weight = loss_weight
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)

        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
        )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y (Tensor): tensors of shape (N,C,H,W)
        Returns: Tensor
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"

        if x.type() != self.gaussian_filter.gaussian_window.type():
            x = x.type_as(self.gaussian_filter.gaussian_window)
        if y.type() != self.gaussian_filter.gaussian_window.type():
            y = y.type_as(self.gaussian_filter.gaussian_window)

        loss = 1 - self.msssim(x, y)

        return self.loss_weight * loss

    def msssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1e-12, 1)
        y = y.clamp(1e-12, 1)

        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)
            ssim = ssim.mean()
            cs = cs.mean()

            if i == 4:
                ms_components.append(ssim ** w)
            else:
                ms_components.append(cs ** w)
                padding = [s % 2 for s in x.shape[2:]]  # spatial padding
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)

        msssim = math.prod(ms_components)  # equ 7 in ref2

        return msssim

    def _ssim(
            self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        a1 = 2 * mu_x * mu_y + self.C1
        a2 = 2 * sigma_xy + self.C2
        b1 = mu_x.pow(2) + mu_y.pow(2) + self.C1
        b2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l = a1 / b1
        cs = a2 / b2
        ssim = l * cs

        return ssim, cs
