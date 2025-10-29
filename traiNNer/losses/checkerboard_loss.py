#!/usr/bin/env python3
"""
CheckerboardLoss: A robust loss function for detecting and penalizing checkerboard artifacts
in super-resolution networks using PixelShuffle upsampling.

This loss specifically addresses the checkerboard pattern problem that occurs when using
PixelShuffle layers in CNN architectures. The artifacts are most visible in flat/smooth
areas of images (sky, walls, skin) where subtle chessboard patterns can appear.

A Refined implementation by Philip Hofmann, based on initial concept/loss code by umzi

Key Features:
- Device-agnostic implementation (works on CPU/GPU)
- Proper loss weight scaling
- Robust parameter validation
- Efficient tensor operations
- Comprehensive documentation

Usage:
    # In config.yaml
    losses:
      - type: CheckerboardLoss
        loss_weight: 1.0
        scale: 4
        criterion: "charbonnier"  # "l1", "l2", or "charbonnier"
        eps: 1e-12

References:
- "Checkerboard artifacts in super-resolution" research
- ICNR initialization as complementary prevention
- ParagonSR architecture usage
"""

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CheckerboardLoss(nn.Module):
    """
    Checkerboard Loss for detecting and penalizing PixelShuffle artifacts.

    This loss works by:
    1. Using PixelUnshuffle to decompose both prediction and target into local blocks
    2. Computing pairwise differences within each scale×scale block
    3. Comparing these differences between prediction and target using a specified criterion

    The intuition is that in flat areas, neighboring pixels should have similar values.
    Checkerboard artifacts break this consistency within local neighborhoods.

    Args:
        loss_weight (float): Weight for this loss component. Default: 1.0
        scale (int): Upscale factor used in the model (must match PixelShuffle). Default: 4
        criterion (str): Loss criterion for comparison. Options: "l1", "l2", "charbonnier".
                        Default: "charbonnier"
        eps (float): Small value for numerical stability (only used with charbonnier). Default: 1e-12

    Note:
        - scale should match the upsampling factor in your network
        - Use criterion="charbonnier" for most robust results
        - Combine with ICNR initialization for best artifact prevention
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        scale: int = 4,
        criterion: Literal["l1", "l2", "charbonnier"] = "charbonnier",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        # Validate parameters
        if loss_weight <= 0:
            raise ValueError(f"loss_weight must be positive, got: {loss_weight}")
        if scale <= 0:
            raise ValueError(f"scale must be a positive integer, got: {scale}")
        if criterion not in ["l1", "l2", "charbonnier"]:
            raise ValueError(
                f"criterion must be one of ['l1', 'l2', 'charbonnier'], got: {criterion}"
            )
        if criterion == "charbonnier" and eps <= 0:
            raise ValueError(
                f"eps must be positive for charbonnier criterion, got: {eps}"
            )

        self.loss_weight = loss_weight
        self.scale = scale
        self.criterion = criterion
        self.eps = eps

        # Initialize PixelUnshuffle for decomposing pixels back into local blocks
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

        # Create upper triangular mask for pairwise comparisons
        # This mask selects unique pairs (i,j) where i < j to avoid double counting
        self.register_triupper_mask()

        # Set up the criterion function
        self._setup_criterion()

    def register_triupper_mask(self) -> None:
        """
        Register the upper triangular mask as a buffer.

        The mask is used to select unique pairwise differences within each local block.
        Shape: (1, 1, scale^2, scale^2, 1, 1) - broadcastable to any input tensor
        """
        # Create mask of shape (scale^2, scale^2) with True for upper triangular part
        triupper_mask = torch.triu(
            torch.ones(self.scale**2, self.scale**2, dtype=torch.bool),
            diagonal=1,
        )

        # Reshape for broadcasting: (1, 1, scale^2, scale^2, 1, 1)
        # The extra dimensions allow easy broadcasting during forward pass
        triupper_mask = triupper_mask.view(1, 1, self.scale**2, self.scale**2, 1, 1)

        # Register as buffer (automatically moved to same device as module)
        self.register_buffer("triupper_mask", triupper_mask)

    def _setup_criterion(self) -> None:
        """Setup the loss criterion based on the specified type."""
        if self.criterion == "l1":
            self.criterion_func = nn.L1Loss(reduction="none")
        elif self.criterion == "l2":
            self.criterion_func = nn.MSELoss(reduction="none")
        else:  # charbonnier

            def charbonnier_criterion(x: Tensor, y: Tensor) -> Tensor:
                return torch.sqrt((x - y) ** 2 + self.eps)

            self.criterion_func = charbonnier_criterion

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute checkerboard loss between prediction and target.

        Args:
            pred (Tensor): Predicted super-resolved image, shape (B, C, H, W)
            target (Tensor): Ground truth high-resolution image, shape (B, C, H, W)

        Returns:
            Tensor: Scalar checkerboard loss value
        """
        # Validate input shapes
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}"
            )
        if pred.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {pred.shape}")

        # Decompose images into local blocks using PixelUnshuffle
        # This reverses the PixelShuffle operation, splitting each scale×scale neighborhood
        pred_unshuffled = self.pixel_unshuffle(pred)  # (B, C*scale^2, H/scale, W/scale)
        target_unshuffled = self.pixel_unshuffle(target)

        # Reshape to separate individual local blocks
        # Now: (B, C, scale^2, H/scale, W/scale) where scale^2 is the pixel positions
        B, C, _, H, W = pred_unshuffled.shape
        pred_groups = pred_unshuffled.view(B, C, -1, H, W)
        target_groups = target_unshuffled.view(B, C, -1, H, W)

        # Compute pairwise differences within each local block
        # Add dimensions for broadcasting: (B, C, 1, scale^2, H, W) and (B, C, scale^2, 1, H, W)
        pred_diffs = pred_groups.unsqueeze(2) - pred_groups.unsqueeze(
            3
        )  # (B, C, scale^2, scale^2, H, W)
        target_diffs = target_groups.unsqueeze(2) - target_groups.unsqueeze(3)

        # Use upper triangular mask to get unique pairs and avoid self-comparisons
        pred_diffs_selected = pred_diffs.masked_select(
            self.triupper_mask
        )  # (B, C, num_pairs, H, W)
        target_diffs_selected = target_diffs.masked_select(self.triupper_mask)

        # Reshape to match criterion input expectations
        pred_diffs_selected = pred_diffs_selected.view(B, C, -1, H, W)
        target_diffs_selected = target_diffs_selected.view(B, C, -1, H, W)

        # Apply the specified criterion to compare differences
        loss_per_element = self.criterion_func(
            pred_diffs_selected, target_diffs_selected
        )

        # Average over all dimensions to get a scalar loss
        loss = loss_per_element.mean()

        # Apply loss weight
        return self.loss_weight * loss


def _test_checkerboard_loss() -> None:
    """
    Basic test function for the CheckerboardLoss module.

    This function tests:
    - Forward pass with different input sizes
    - Device compatibility (CPU/GPU if available)
    - Different criterion types
    - Gradient computation

    Run with: python -m traiNNer.losses.checkerboard_loss
    """
    print("Testing CheckerboardLoss implementation...")

    # Test parameters
    batch_size, channels, height, width = 2, 3, 64, 64
    scale = 4

    # Create test tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = torch.randn(
        batch_size, channels, height * scale, width * scale, device=device
    )
    target = torch.randn_like(pred)

    # Test different criteria by creating separate loss modules
    print("\nTesting with criterion: l1")
    loss_fn_l1 = CheckerboardLoss(loss_weight=1.0, scale=scale, criterion="l1")
    loss_fn_l1.to(device)
    loss_val = loss_fn_l1(pred, target)
    print(f"Loss value: {loss_val.item():.6f}")
    loss_val.backward()
    print(f"Gradient computation: {'✓' if pred.grad is not None else '✗'}")
    pred.grad.zero_()
    target.grad.zero_()

    print("\nTesting with criterion: l2")
    loss_fn_l2 = CheckerboardLoss(loss_weight=1.0, scale=scale, criterion="l2")
    loss_fn_l2.to(device)
    loss_val = loss_fn_l2(pred, target)
    print(f"Loss value: {loss_val.item():.6f}")
    loss_val.backward()
    print(f"Gradient computation: {'✓' if pred.grad is not None else '✗'}")
    pred.grad.zero_()
    target.grad.zero_()

    print("\nTesting with criterion: charbonnier")
    loss_fn_charbonnier = CheckerboardLoss(
        loss_weight=1.0, scale=scale, criterion="charbonnier"
    )
    loss_fn_charbonnier.to(device)
    loss_val = loss_fn_charbonnier(pred, target)
    print(f"Loss value: {loss_val.item():.6f}")
    loss_val.backward()
    print(f"Gradient computation: {'✓' if pred.grad is not None else '✗'}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    _test_checkerboard_loss()
