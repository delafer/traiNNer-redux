from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FeatureMatchingLoss(nn.Module):
    """
    Encourages generator to match discriminator's intermediate features.

    This loss helps stabilize GAN training by encouraging the generator to produce
    features that the discriminator finds similar to real images, especially useful
    for multi-branch discriminators like MUNet.

    Config options:
      - loss_weight: global multiplier
      - reduction: 'mean' or 'sum' for loss aggregation
      - layers: optional list of layer indices to use for feature matching
      - criterion: 'l1', 'l2', or 'charbonnier' for feature difference
      - eps: small constant for numerical stability (charbonnier criterion)
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        layers: list[int] | None = None,
        criterion: Literal["l1", "l2", "charbonnier"] = "l1",
        eps: float = 1e-6,
        **_: dict,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        if criterion not in ("l1", "l2", "charbonnier"):
            raise ValueError("criterion must be 'l1', 'l2', or 'charbonnier'")

        self.reduction = reduction
        self.layers = layers
        self.criterion = criterion
        self.eps = eps

    def forward(
        self, disc_real_feats: list[torch.Tensor], disc_fake_feats: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feature matching loss between real and fake discriminator features.

        Args:
            disc_real_feats: List of intermediate features from discriminator on real images
            disc_fake_feats: List of intermediate features from discriminator on fake images

        Returns:
            Feature matching loss value
        """
        if len(disc_real_feats) != len(disc_fake_feats):
            raise ValueError(
                f"Feature list length mismatch: {len(disc_real_feats)} vs {len(disc_fake_feats)}"
            )

        # Select specific layers if specified
        if self.layers is not None:
            selected_real_feats = []
            selected_fake_feats = []
            for layer_idx in self.layers:
                if 0 <= layer_idx < len(disc_real_feats):
                    selected_real_feats.append(disc_real_feats[layer_idx])
                    selected_fake_feats.append(disc_fake_feats[layer_idx])
            disc_real_feats = selected_real_feats
            disc_fake_feats = selected_fake_feats

        if not disc_real_feats:
            return torch.tensor(0.0, device="cpu")

        loss = 0.0
        count = 0

        for fr, ff in zip(disc_real_feats, disc_fake_feats, strict=False):
            if fr.shape != ff.shape:
                # Resize features to match if needed
                if fr.numel() > ff.numel():
                    ff = torch.nn.functional.interpolate(
                        ff, size=fr.shape[2:], mode="bilinear", align_corners=False
                    )
                else:
                    fr = torch.nn.functional.interpolate(
                        fr, size=ff.shape[2:], mode="bilinear", align_corners=False
                    )

            # Compute feature difference based on criterion
            diff = fr.detach() - ff

            if self.criterion == "l1":
                element_loss = diff.abs().mean()
            elif self.criterion == "l2":
                element_loss = (diff * diff).mean()
            else:  # charbonnier
                element_loss = torch.sqrt(diff * diff + self.eps).mean()

            loss += element_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=disc_real_feats[0].device)

        if self.reduction == "mean":
            return loss / count
        else:
            return loss
