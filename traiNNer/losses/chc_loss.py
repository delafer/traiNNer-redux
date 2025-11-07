from __future__ import annotations

import torch
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CHCLoss(nn.Module):
    r"""Clipped pseudo-Huber with Cosine Similarity (CHC) loss.

    Combines a robust L1/H\* penalty with a cosine-similarity term and clamps the
    resulting loss to suppress well-converged or noisy pixels.

    References:
      - AIM2020 Real-SR Challenge
      - HDR ExpandNet
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        criterion: str = "huber",
        loss_lambda: float = 0.0,
        clip_min: float = 1.0 / 255.0,
        clip_max: float = 254.0 / 255.0,
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "Unsupported reduction mode: {reduction}. "
                "Supported options are: 'none', 'mean', 'sum'."
            )

        if criterion not in {"l1", "huber"}:
            raise ValueError(
                f"Unsupported CHC criterion '{criterion}'. Use 'l1' or 'huber'."
            )

        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.criterion = criterion
        self.loss_lambda = float(loss_lambda)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)

    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        cosine_term = (1.0 - self.similarity(pred, target)).mean()

        if self.criterion == "l1":
            base = torch.abs(pred - target)
        else:  # pseudo-huber / charbonnier
            base = torch.sqrt((pred - target) ** 2 + 1e-12)

        combined = base + self.loss_lambda * cosine_term
        clipped = torch.clamp(combined, self.clip_min, self.clip_max)

        if self.reduction == "mean":
            loss = clipped.mean()
        elif self.reduction == "sum":
            loss = clipped.sum()
        else:
            loss = clipped

        return loss * self.loss_weight
