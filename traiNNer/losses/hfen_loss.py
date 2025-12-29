from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import LOSS_REGISTRY


def _make_log_kernel(
    kernel_size: int = 7, sigma: float = 1.0, device=None, dtype=torch.float32
):
    """Create a 2D Laplacian of Gaussian kernel (LoG)."""
    assert kernel_size % 2 == 1 and kernel_size >= 3
    k = kernel_size // 2
    xs = torch.arange(-k, k + 1, dtype=dtype, device=device)
    ys = xs.view(-1, 1)
    xx = xs.unsqueeze(0).expand(kernel_size, kernel_size)
    yy = ys.expand(kernel_size, kernel_size)
    r2 = xx**2 + yy**2
    sigma2 = sigma * sigma
    # LoG kernel formula
    factor = (r2 - 2 * sigma2) / (sigma2**2)
    kernel = factor * torch.exp(-r2 / (2 * sigma2))
    kernel = kernel - kernel.mean()
    return kernel


@LOSS_REGISTRY.register()
class HFENLoss(nn.Module):
    """
    High-Frequency Error Norm (HFEN) loss using a Laplacian-of-Gaussian filter.
    Computes difference between HF maps of prediction and target.

    Config options:
      - loss_weight: global multiplier
      - kernel_size: odd int, LoG kernel size (7 is common)
      - sigma: LoG sigma
      - reduction: 'mean' or 'sum'
      - eps: small const for robust charb/avoid div by zero
      - criterion: 'l2' | 'l1' | 'charbonnier'
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        kernel_size: int = 7,
        sigma: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        eps: float = 1e-6,
        criterion: Literal["l2", "l1", "charbonnier"] = "charbonnier",
        **_: dict,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.k = int(kernel_size)
        self.sigma = float(sigma)
        self.reduction = reduction
        self.eps = float(eps)
        self.criterion = criterion

        # create kernel placeholder; actual device set at forward
        self.register_buffer("kernel_placeholder", torch.zeros(1))

    def _get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        kernel = _make_log_kernel(self.k, self.sigma, device=device, dtype=dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        return kernel

    def _apply_log(self, img: torch.Tensor) -> torch.Tensor:
        # assume img shape (B,C,H,W) in [0,1] or similar
        # apply LoG per-channel by grouped conv
        _B, C, _H, _W = img.shape
        kernel = self._get_kernel(img)
        kernel = kernel.repeat(C, 1, 1, 1)  # (C,1,k,k)
        padding = self.k // 2
        # conv2d with groups=C
        hf = F.conv2d(img, kernel, bias=None, stride=1, padding=padding, groups=C)
        return hf

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("HFEN expects pred and target same shape")
        # compute high-frequency maps
        hf_pred = self._apply_log(pred)
        hf_tgt = self._apply_log(target)
        diff = hf_pred - hf_tgt
        if self.criterion == "l2":
            element = diff * diff
        elif self.criterion == "l1":
            element = diff.abs()
        else:  # charbonnier
            element = torch.sqrt(diff * diff + self.eps)
        if self.reduction == "mean":
            loss = element.mean()
        else:
            loss = element.sum()
        return self.loss_weight * loss
