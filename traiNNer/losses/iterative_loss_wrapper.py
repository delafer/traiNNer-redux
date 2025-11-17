#!/usr/bin/env python3
"""
Iterative Loss Wrapper for traiNNer-redux
Author: Philip Hofmann

A sophisticated loss wrapper that enables iteration-based loss weight scheduling,
allowing for dynamic loss activation, ramping, and deactivation throughout training.

Features:
- start_iter: When to begin applying the loss
- target_iter: When to reach target_weight
- target_weight: Final weight to ramp to
- disable_after: When to completely disable the loss
- Smooth weight transitions for training stability

Licensed under the MIT License.
"""

import warnings
from typing import Any, Dict, Union

import torch
from torch import nn


class IterativeLossWrapper(nn.Module):
    """
    A wrapper that enables iteration-based loss weight scheduling.

    This wrapper allows loss weights to change dynamically based on training iteration,
    enabling sophisticated training strategies like:
    - Gradual activation of perceptual losses
    - GAN loss ramp-up schedules
    - Phase-based training approaches
    - Artifact suppression removal in later stages

    Args:
        loss_module: The underlying loss function
        loss_weight: Initial/base loss weight
        start_iter: When to start applying this loss (default: 0)
        target_iter: When to reach target_weight (default: start_iter)
        target_weight: Final weight to ramp to (default: loss_weight)
        disable_after: When to completely disable the loss (default: None)
        schedule_type: Type of scheduling ('linear', 'cosine', 'step') (default: 'linear')
        warn_on_unused: Warn if loss becomes unused (default: True)
    """

    def __init__(
        self,
        loss_module: nn.Module,
        loss_weight: float = 1.0,
        start_iter: int = 0,
        target_iter: int | None = None,
        target_weight: float | None = None,
        disable_after: int | None = None,
        schedule_type: str = "linear",
        warn_on_unused: bool = True,
    ) -> None:
        super().__init__()

        self.loss_module = loss_module
        self.base_loss_weight = loss_weight
        self.start_iter = start_iter
        self.target_iter = target_iter if target_iter is not None else start_iter
        self.target_weight = target_weight if target_weight is not None else loss_weight
        self.disable_after = disable_after
        self.schedule_type = schedule_type.lower()
        self.warn_on_unused = warn_on_unused

        # Store the original loss attributes for compatibility
        self.loss_weight = loss_weight

        # Validate parameters
        if self.target_iter < self.start_iter:
            raise ValueError(
                f"target_iter ({self.target_iter}) must be >= start_iter ({self.start_iter})"
            )

        if self.target_weight < 0:
            raise ValueError(
                f"target_weight ({self.target_weight}) must be non-negative"
            )

        self._last_effective_weight = 0.0
        self._has_warned_unused = False

    def _calculate_schedule_weight(self, current_iter: int) -> float:
        """
        Calculate the weight based on the current iteration and schedule type.

        Args:
            current_iter: Current training iteration

        Returns:
            Effective weight for this iteration
        """
        # Check if loss should be disabled
        if self.disable_after is not None and current_iter >= self.disable_after:
            return 0.0

        # Check if loss hasn't started yet
        if current_iter < self.start_iter:
            return 0.0

        # If no ramp needed, return base weight
        if self.target_iter <= self.start_iter:
            return self.target_weight

        # Calculate progress through the ramp period
        ramp_progress = (current_iter - self.start_iter) / (
            self.target_iter - self.start_iter
        )
        ramp_progress = max(0.0, min(1.0, ramp_progress))  # Clamp to [0, 1]

        if self.schedule_type == "linear":
            # Linear interpolation from base_weight to target_weight
            effective_weight = (
                self.base_loss_weight
                + (self.target_weight - self.base_loss_weight) * ramp_progress
            )

        elif self.schedule_type == "cosine":
            # Cosine easing for smoother transitions
            import math

            eased_progress = 0.5 * (1 - math.cos(math.pi * ramp_progress))
            effective_weight = (
                self.base_loss_weight
                + (self.target_weight - self.base_loss_weight) * eased_progress
            )

        elif self.schedule_type == "step":
            # Step function: jump to target_weight at target_iter
            effective_weight = (
                self.target_weight
                if current_iter >= self.target_iter
                else self.base_loss_weight
            )

        else:
            warnings.warn(
                f"Unknown schedule_type '{self.schedule_type}', using linear",
                stacklevel=2,
            )
            effective_weight = (
                self.base_loss_weight
                + (self.target_weight - self.base_loss_weight) * ramp_progress
            )

        return effective_weight

    def forward(self, *args, current_iter: int | None = None, **kwargs) -> Any:
        """
        Forward pass with iteration-aware weight calculation.

        Args:
            *args: Arguments passed to the underlying loss
            current_iter: Current training iteration (required)
            **kwargs: Keyword arguments passed to the underlying loss

        Returns:
            Loss value with effective weight applied
        """
        if current_iter is None:
            raise ValueError("current_iter must be provided to IterativeLossWrapper")

        # Calculate effective weight for this iteration
        effective_weight = self._calculate_schedule_weight(current_iter)
        self._last_effective_weight = effective_weight

        # Warn if loss is effectively unused (very small weight)
        if (
            self.warn_on_unused
            and effective_weight < 1e-6
            and not self._has_warned_unused
        ):
            warnings.warn(
                f"Loss {self.loss_module.__class__.__name__} has very small effective weight "
                f"({effective_weight:.2e}) at iteration {current_iter}. "
                f"This may indicate configuration issues.",
                UserWarning,
                stacklevel=3,
            )
            self._has_warned_unused = True

        # Compute the base loss
        if effective_weight == 0.0:
            # Return zero loss to avoid computation overhead
            base_loss = self.loss_module(*args, **kwargs)
            if isinstance(base_loss, dict):
                return {k: v * 0.0 for k, v in base_loss.items()}
            else:
                device = args[0].device if args else torch.device("cpu")
                return torch.tensor(
                    0.0,
                    device=device,
                    dtype=base_loss.dtype
                    if hasattr(base_loss, "dtype")
                    else torch.float32,
                )

        # Compute the base loss with original weight, then apply effective weight
        base_loss = self.loss_module(*args, **kwargs)

        # Apply effective weight
        if isinstance(base_loss, dict):
            # Handle losses that return multiple components
            weighted_loss = {}
            for key, value in base_loss.items():
                weighted_loss[key] = value * effective_weight
            return weighted_loss
        else:
            return base_loss * effective_weight

    def get_current_weight(self, current_iter: int) -> float:
        """Get the current effective weight without computing loss."""
        return self._calculate_schedule_weight(current_iter)

    def is_active(self, current_iter: int) -> bool:
        """Check if the loss is currently active (has non-zero weight)."""
        return self._calculate_schedule_weight(current_iter) > 1e-6

    def get_schedule_info(self) -> dict[str, Any]:
        """Get information about the current schedule."""
        return {
            "base_loss_weight": self.base_loss_weight,
            "start_iter": self.start_iter,
            "target_iter": self.target_iter,
            "target_weight": self.target_weight,
            "disable_after": self.disable_after,
            "schedule_type": self.schedule_type,
        }

    def __repr__(self) -> str:
        schedule_info = self.get_schedule_info()
        return f"IterativeLossWrapper({self.loss_module.__class__.__name__}, {schedule_info})"


# Utility function to create scheduled losses
def create_iterative_loss(
    loss_module: nn.Module, loss_config: dict[str, Any]
) -> IterativeLossWrapper:
    """
    Create an IterativeLossWrapper from a loss configuration.

    Args:
        loss_module: The underlying loss function
        loss_config: Loss configuration dict containing scheduling parameters

    Returns:
        Configured IterativeLossWrapper
    """
    # Extract scheduling parameters
    schedule_params = {}

    for param in [
        "start_iter",
        "target_iter",
        "target_weight",
        "disable_after",
        "schedule_type",
        "warn_on_unused",
    ]:
        if param in loss_config:
            schedule_params[param] = loss_config[param]

    # Default values
    if "start_iter" not in schedule_params:
        schedule_params["start_iter"] = 0
    if "schedule_type" not in schedule_params:
        schedule_params["schedule_type"] = "linear"
    if "warn_on_unused" not in schedule_params:
        schedule_params["warn_on_unused"] = True

    return IterativeLossWrapper(
        loss_module=loss_module,
        loss_weight=loss_config["loss_weight"],
        **schedule_params,
    )
