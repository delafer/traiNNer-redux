from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    """
    Isotropic Total Variation loss.

    Encourages spatial smoothness by penalizing local gradients, which helps
    suppress ringing, speckle noise, and tiny oscillations. At a very small
    weight it preserves real edges and textures.

    TV(x) = mean( sqrt( (dx)^2 + (dy)^2 + eps ) )

    Notes:
    - This implementation is integrated with traiNNer-redux's LOSS_REGISTRY.
    - `loss_weight` is handled inside the module to match the existing config style.
    - Typical usage for clean pretrain:
        type: tvloss
        loss_weight: 0.005–0.02
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        eps: float = 1e-6,
        **_: dict,
    ) -> None:
        """
        Args:
            loss_weight: Global multiplier applied to the TV term.
            reduction: 'mean' or 'sum' over all pixels and channels.
            eps: Small constant for numerical stability.
            **_: Extra kwargs from YAML (ignored for forward-compatibility).
        """
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute TV loss on the prediction.

        Note:
        - `target` is accepted for compatibility with the generic SRModel loss
          calling convention (loss(pred, target)). It is ignored here because
          TV is defined purely on the prediction.
        """
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]

        dx_pad = torch.zeros_like(x[:, :, :, :1])
        dy_pad = torch.zeros_like(x[:, :, :1, :])

        dx = torch.cat([dx, dx_pad], dim=3)
        dy = torch.cat([dy, dy_pad], dim=2)

        tv_map = torch.sqrt(dx * dx + dy * dy + self.eps)

        if self.reduction == "mean":
            tv = tv_map.mean()
        else:
            tv = tv_map.sum()

        return self.loss_weight * tv
