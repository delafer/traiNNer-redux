from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class AdaptiveBlockTVLoss(nn.Module):
    """
    Adaptive, block-aware Total Variation loss.

    KEY IDEA:
    ----------
    TV encourages local smoothness (good for damping ringing/noise),
    but destroys texture because it is *blind*.

    This version:
        - detects subpixel/checkerboard inconsistencies
        - increases TV weight where block differences are high
        - decreases TV weight where structure is coherent
        - preserves real edges by modulating TV with block-consistency map

    Smooth where the generator is "wrong".
    Preserve where it is "right".

    PARAMETERS:
    -----------
    loss_weight: a global multiplier (like your existing code)
    block_size: size of lattice to measure inconsistency (2 or 4 typical)
    sharpness: controls sensitivity of reweighting (higher = more selective)
    eps: numerical stability
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        block_size: int = 2,
        sharpness: float = 4.0,
        reduction: Literal["mean", "sum"] = "mean",
        eps: float = 1e-6,
        **_: dict,
    ) -> None:
        super().__init__()

        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        if block_size < 2:
            raise ValueError("block_size must be >= 2")

        self.loss_weight = float(loss_weight)
        self.block_size = int(block_size)
        self.sharpness = float(sharpness)
        self.reduction = reduction
        self.eps = eps

    def _compute_checkerboard_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        Measures block inconsistency.

        For each px, compute how different its value is from the average value
        of its block. Large inconsistency = likely checkerboard/aliasing artifact.
        """

        B = self.block_size

        # --- Compute block mean ---
        # reshape into (b, c, h//B, B, w//B, B)
        h, w = x.shape[2], x.shape[3]
        H = (h // B) * B
        W = (w // B) * B

        # Safe crop to multiple of block size
        xc = x[:, :, :H, :W]

        # Reshape into block grid
        blk = xc.reshape(x.size(0), x.size(1), H // B, B, W // B, B)

        # Block mean over the B×B pixels
        blk_mean = blk.mean(dim=(3, 5), keepdim=True)

        # Broadcast difference
        diff = (blk - blk_mean).abs()

        # Map back to image shape
        # (b, c, H//B, B, W//B, B) → (b, c, H, W)
        diff_map = diff.reshape(x.size(0), x.size(1), H, W)

        # Pad borders back if needed
        if diff_map.shape[2:] != x.shape[2:]:
            diff_map = torch.nn.functional.pad(
                diff_map,
                (0, x.shape[3] - diff_map.shape[3], 0, x.shape[2] - diff_map.shape[2]),
            )

        # Normalize
        norm_diff = diff_map / (diff_map.mean() + float(self.eps))

        # Convert to a reweighting map:
        # high inconsistency → high weight
        # smooth exponential control
        weight = torch.sigmoid(self.sharpness * norm_diff)

        return weight

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        # TV gradients
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]

        # Padding for consistent shapes
        dx_pad = torch.zeros_like(x[:, :, :, :1])
        dy_pad = torch.zeros_like(x[:, :, :1, :])
        dx = torch.cat([dx, dx_pad], dim=3)
        dy = torch.cat([dy, dy_pad], dim=2)

        tv_map = torch.sqrt(dx * dx + dy * dy + float(self.eps))

        # Compute adaptive weights
        weight = self._compute_checkerboard_weight(x)

        # Apply weight to TV
        weighted_tv = tv_map * weight

        if self.reduction == "mean":
            tv = weighted_tv.mean()
        else:
            tv = weighted_tv.sum()

        return self.loss_weight * tv
