import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class LaplacianPyramidLoss(nn.Module):
    """Laplacian Pyramid Loss for Super-Resolution.

    This loss computes the difference between Laplacian pyramids of the
    super-resolved and ground truth images, encouraging the model to
    preserve structural details at multiple scales.
    """

    def __init__(
        self, loss_weight: float = 1.0, levels: int = 4, criterion: str = "charbonnier"
    ) -> None:
        """
        Initialize the LaplacianPyramidLoss.

        Args:
            loss_weight (float): Weight for the loss. Default: 1.0
            levels (int): Number of pyramid levels. Default: 4
            criterion (str): Criterion for loss computation ('l1', 'l2', 'charbonnier'). Default: 'charbonnier'
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.levels = levels

        if criterion == "l1":
            self.criterion = nn.L1Loss(reduction="mean")
        elif criterion == "l2":
            self.criterion = nn.MSELoss(reduction="mean")
        elif criterion == "charbonnier":
            self.criterion = lambda x, y: charbonnier_loss(
                x, y, reduction="mean"
            ).mean()
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

    def build_laplacian_pyramid(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Build a Laplacian pyramid from an input image."""
        pyramid = []
        current = img

        # Build Gaussian pyramid
        gaussian_pyramid = [current]
        for i in range(self.levels):
            # Downsample
            downsampled = F.interpolate(
                current, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            gaussian_pyramid.append(downsampled)
            current = downsampled

        # Build Laplacian pyramid
        for i in range(self.levels):
            # Upsample the next level
            upsampled = F.interpolate(
                gaussian_pyramid[i + 1],
                size=gaussian_pyramid[i].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            # Laplacian is difference between current level and upsampled next level
            laplacian = gaussian_pyramid[i] - upsampled
            pyramid.append(laplacian)

        # Add the coarsest level
        pyramid.append(gaussian_pyramid[-1])

        return pyramid

    def forward(self, sr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Laplacian pyramid loss.

        Args:
            sr (torch.Tensor): Super-resolved images of shape (N, C, H, W)
            gt (torch.Tensor): Ground truth images of shape (N, C, H, W)

        Returns:
            torch.Tensor: Computed Laplacian pyramid loss
        """
        # Build Laplacian pyramids
        sr_pyramid = self.build_laplacian_pyramid(sr)
        gt_pyramid = self.build_laplacian_pyramid(gt)

        # Compute loss for each level (excluding the coarsest level)
        loss = 0.0
        for i in range(self.levels):
            level_loss = self.criterion(sr_pyramid[i], gt_pyramid[i])
            loss += level_loss

        return self.loss_weight * loss
