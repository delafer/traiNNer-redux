from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from traiNNer.utils import get_root_logger
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
        layers: list[int | str] | None = None,
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
        self.logger = get_root_logger()

    def forward(
        self,
        disc_real_feats: list[torch.Tensor],
        disc_fake_feats: list[torch.Tensor],
        **kwargs,
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
            for layer_ref in self.layers:
                layer_idx = None

                # Handle string layer names (e.g., 'down1', 'down2', 'mid')
                if isinstance(layer_ref, str):
                    # Map string names to indices
                    layer_map = {
                        "down1": 1,  # first down block output
                        "down2": 2,  # second down block output
                        "mid": -1,  # midpoint/bottleneck
                    }
                    layer_idx = layer_map.get(layer_ref)
                    if layer_idx is None:
                        self.logger.warning(
                            f"Unknown layer name '{layer_ref}', ignoring"
                        )
                        continue

                # Handle integer indices
                elif isinstance(layer_ref, int):
                    layer_idx = layer_ref

                # Check bounds and add to selection
                if layer_idx is not None and 0 <= layer_idx < len(disc_real_feats):
                    selected_real_feats.append(disc_real_feats[layer_idx])
                    selected_fake_feats.append(disc_fake_feats[layer_idx])
                elif layer_idx is not None and -len(disc_real_feats) <= layer_idx < 0:
                    # Handle negative indexing
                    selected_real_feats.append(disc_real_feats[layer_idx])
                    selected_fake_feats.append(disc_fake_feats[layer_idx])

            disc_real_feats = selected_real_feats
            disc_fake_feats = selected_fake_feats

        if not disc_real_feats:
            # Return a zero tensor on the same device as input features to maintain gradients
            if disc_fake_feats:
                return torch.zeros_like(disc_fake_feats[0]).mean()
            else:
                return torch.tensor(0.0, requires_grad=True)

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
            # Return zero loss maintaining gradient flow
            return torch.zeros_like(
                disc_real_feats[0], dtype=disc_real_feats[0].dtype
            ).mean()

        result = loss / count if self.reduction == "mean" else loss
        return result
