#!/usr/bin/env python3
"""
Dynamic Loss Scheduling Module for traiNNer-redux
Author: Philip Hofmann

A sophisticated loss scheduling system that automatically adjusts loss weights
based on current loss values during training. This prevents loss dominance,
maintains training balance, and adapts to changing dynamics automatically.

Features:
- Momentum-based smoothing for stable adaptation
- Automatic loss balance tracking
- Safety bounds to prevent extreme weight changes
- Graceful degradation when losses are stable
- Support for both scalar and dict-based losses
- Easy integration with existing loss framework

Key Parameters:
- momentum (float): Exponential smoothing factor for loss tracking
- adaptation_rate (float): How quickly weights adapt to loss changes
- min_weight (float): Minimum loss weight multiplier
- max_weight (float): Maximum loss weight multiplier
- adaptation_threshold (float): Minimum loss change to trigger adaptation
- baseline_iterations (int): Iterations to establish baseline before adapting

Licensed under the MIT License.
"""

import math
from typing import Any, Dict, Union

import torch
from torch import nn


class DynamicLossScheduler(nn.Module):
    """
    A dynamic loss scheduling system that automatically adjusts loss weights
    based on current loss values to maintain optimal training balance.

    This system monitors loss magnitudes and adapts weights to prevent:
    - One loss overwhelming others
    - Training instabilities from unbalanced gradients
    - Poor convergence from static weight configurations

    Features:
    - Exponential smoothing for stable tracking
    - Adaptive weight bounds with safety limits
    - Loss imbalance detection and correction
    - Graceful handling of both scalar and dict losses
    - Comprehensive logging for debugging and monitoring
    """

    def __init__(
        self,
        base_weights: dict[str, float],
        momentum: float = 0.9,
        adaptation_rate: float = 0.01,
        min_weight: float = 1e-6,
        max_weight: float = 100.0,
        adaptation_threshold: float = 0.1,
        baseline_iterations: int = 100,
        enable_monitoring: bool = True,
    ) -> None:
        """
        Initialize the dynamic loss scheduler.

        Args:
            base_weights: Dictionary of base loss weights for each loss type
            momentum: Exponential smoothing factor (0.0-1.0) for loss tracking
            adaptation_rate: Rate of weight adaptation per iteration (0.001-0.1)
            min_weight: Minimum possible weight multiplier
            max_weight: Maximum possible weight multiplier
            adaptation_threshold: Minimum relative loss change to trigger adaptation
            baseline_iterations: Number of iterations to establish baseline before adapting
            enable_monitoring: Enable detailed monitoring and logging
        """
        super().__init__()

        # Validate parameters
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Momentum must be between 0.0 and 1.0, got {momentum}")
        if not 0.0 < adaptation_rate <= 1.0:
            raise ValueError(f"Adaptation rate must be positive, got {adaptation_rate}")
        if not 0.0 <= min_weight <= max_weight:
            raise ValueError(
                f"Invalid weight bounds: min={min_weight}, max={max_weight}"
            )
        if baseline_iterations < 0:
            raise ValueError(
                f"Baseline iterations must be non-negative, got {baseline_iterations}"
            )

        self.base_weights = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(v)) for k, v in base_weights.items()}
        )
        self.momentum = momentum
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adaptation_threshold = adaptation_threshold
        self.baseline_iterations = baseline_iterations
        self.enable_monitoring = enable_monitoring

        # State tracking variables
        self.register_buffer("_iteration", torch.tensor(0.0))
        self.register_buffer("_baseline_established", torch.tensor(False))

        # Loss tracking with exponential smoothing (using regular dicts for buffers)
        self.smoothed_losses = {}
        self.loss_velocities = {}
        self.current_adjustments = {}
        self.adaptation_history = {}

        # Initialize tracking dictionaries with zeros (using buffers, not parameters)
        for loss_name in base_weights.keys():
            self.register_buffer(f"smoothed_losses_{loss_name}", torch.tensor(0.0))
            self.register_buffer(f"loss_velocities_{loss_name}", torch.tensor(0.0))
            self.register_buffer(f"current_adjustments_{loss_name}", torch.tensor(1.0))
            # Store references for easier access
            self.smoothed_losses[loss_name] = getattr(
                self, f"smoothed_losses_{loss_name}"
            )
            self.loss_velocities[loss_name] = getattr(
                self, f"loss_velocities_{loss_name}"
            )
            self.current_adjustments[loss_name] = getattr(
                self, f"current_adjustments_{loss_name}"
            )
            self.adaptation_history[loss_name] = []

        # Statistics for monitoring
        self.adaptation_count = 0
        self.last_weight_changes = {}

    def forward(
        self,
        current_losses: dict[str, float | torch.Tensor | dict[str, torch.Tensor]],
        current_iter: int,
    ) -> dict[str, float]:
        """
        Update loss weights based on current loss values.

        Args:
            current_losses: Dictionary of current loss values for each loss type
            current_iter: Current training iteration

        Returns:
            Dictionary of adjusted weight multipliers for each loss type
        """
        # Convert input losses to scalar values
        scalar_losses = self._extract_scalar_losses(current_losses)

        # Update iteration counter
        self._iteration.fill_(current_iter)

        # Establish baseline if not done yet
        if not self._baseline_established.item():
            self._establish_baseline(scalar_losses, current_iter)
            # Return float values for all losses
            result = {}
            for loss_name in self.base_weights.keys():
                adjustment_attr = f"current_adjustments_{loss_name}"
                if hasattr(self, adjustment_attr):
                    result[loss_name] = getattr(self, adjustment_attr).item()
                else:
                    result[loss_name] = 1.0
            return result

        # Update smoothed loss values and compute velocities
        self._update_loss_tracking(scalar_losses, current_iter)

        # Compute weight adjustments based on loss dynamics
        adjustments = self._compute_weight_adjustments(scalar_losses)

        # Apply bounds and update adjustment history
        bounded_adjustments = self._apply_bounds(adjustments)
        self._update_adjustment_history(bounded_adjustments)

        # Return float values for all losses
        result = {}
        for loss_name in self.base_weights.keys():
            if loss_name in bounded_adjustments:
                result[loss_name] = bounded_adjustments[loss_name]
            else:
                adjustment_attr = f"current_adjustments_{loss_name}"
                if hasattr(self, adjustment_attr):
                    result[loss_name] = getattr(self, adjustment_attr).item()
                else:
                    result[loss_name] = 1.0

        return result

    def _extract_scalar_losses(
        self, current_losses: dict[str, Any]
    ) -> dict[str, float]:
        """Convert complex loss structures to scalar values for analysis."""
        scalar_losses = {}

        for loss_name, loss_value in current_losses.items():
            if isinstance(loss_value, dict):
                # For multi-component losses, use weighted sum
                scalar_losses[loss_name] = sum(
                    abs(v) if isinstance(v, torch.Tensor) else abs(float(v))
                    for v in loss_value.values()
                )
            elif isinstance(loss_value, torch.Tensor):
                # For tensor losses, take mean and absolute value
                scalar_losses[loss_name] = float(loss_value.mean().abs())
            else:
                # For scalar losses, just take absolute value
                scalar_losses[loss_name] = float(abs(loss_value))

        return scalar_losses

    def _establish_baseline(
        self, scalar_losses: dict[str, float], current_iter: int
    ) -> None:
        """Establish baseline loss values over initial iterations."""
        for loss_name, loss_value in scalar_losses.items():
            smoothed_attr = f"smoothed_losses_{loss_name}"
            if hasattr(self, smoothed_attr):
                # Initialize smoothed loss with first observed value
                getattr(self, smoothed_attr).fill_(loss_value)

        # Mark baseline as established after sufficient iterations
        if current_iter >= self.baseline_iterations:
            self._baseline_established.fill_(True)

            if self.enable_monitoring:
                logger = _get_logger()
                baseline_values = {}
                for loss_name in self.base_weights.keys():
                    smoothed_attr = f"smoothed_losses_{loss_name}"
                    if hasattr(self, smoothed_attr):
                        baseline_values[loss_name] = getattr(self, smoothed_attr).item()
                logger.info(
                    f"Dynamic loss scheduler baseline established at iteration {current_iter}. "
                    f"Baseline loss values: {baseline_values}"
                )

    def _update_loss_tracking(
        self, scalar_losses: dict[str, float], current_iter: int
    ) -> None:
        """Update smoothed loss values and compute loss velocities."""
        dt = 1.0  # Time step between iterations

        for loss_name, current_loss in scalar_losses.items():
            smoothed_attr = f"smoothed_losses_{loss_name}"
            velocity_attr = f"loss_velocities_{loss_name}"

            if not (hasattr(self, smoothed_attr) and hasattr(self, velocity_attr)):
                continue

            prev_smoothed = getattr(self, smoothed_attr).clone()
            prev_velocity = getattr(self, velocity_attr).clone()

            # Exponential smoothing for loss tracking
            alpha = 1.0 - math.exp(-dt / 10.0)  # 10-iteration time constant
            smoothed_loss = alpha * current_loss + (1 - alpha) * prev_smoothed
            getattr(self, smoothed_attr).fill_(smoothed_loss)

            # Compute loss velocity (rate of change)
            loss_change = smoothed_loss - prev_smoothed
            velocity = self.momentum * prev_velocity + (1 - self.momentum) * (
                loss_change / dt
            )
            getattr(self, velocity_attr).fill_(velocity)

    def _compute_weight_adjustments(
        self, scalar_losses: dict[str, float]
    ) -> dict[str, float]:
        """Compute weight adjustments based on current loss dynamics."""
        adjustments = {}

        for loss_name, current_loss in scalar_losses.items():
            if loss_name not in self.base_weights:
                adjustments[loss_name] = 1.0
                continue

            smoothed_attr = f"smoothed_losses_{loss_name}"
            velocity_attr = f"loss_velocities_{loss_name}"

            if not (hasattr(self, smoothed_attr) and hasattr(self, velocity_attr)):
                adjustments[loss_name] = 1.0
                continue

            base_loss = getattr(self, smoothed_attr).item()
            velocity = getattr(self, velocity_attr).item()

            # Skip adaptation if baseline is not established or loss is zero
            if not self._baseline_established.item() or base_loss <= 1e-8:
                adjustments[loss_name] = 1.0
                continue

            # Compute relative loss change
            relative_change = (current_loss - base_loss) / base_loss

            # Determine adjustment based on loss behavior
            adjustment = 1.0  # Default: no change

            # If loss is changing rapidly, adapt weight
            if abs(velocity) > self.adaptation_threshold:
                if velocity > 0:
                    # Loss increasing - reduce weight slightly to stabilize
                    reduction_factor = 1.0 - self.adaptation_rate * min(
                        2.0, abs(relative_change)
                    )
                    adjustment = max(0.1, reduction_factor)
                else:
                    # Loss decreasing - increase weight slightly to encourage progress
                    boost_factor = 1.0 + self.adaptation_rate * min(
                        2.0, abs(relative_change)
                    )
                    adjustment = min(10.0, boost_factor)
            elif abs(relative_change) > self.adaptation_threshold:
                # Static but significantly different from baseline
                if relative_change > 0:
                    # Higher than baseline - reduce weight
                    adjustment = max(0.1, 1.0 - self.adaptation_rate * relative_change)
                else:
                    # Lower than baseline - increase weight slightly
                    adjustment = min(
                        10.0, 1.0 + self.adaptation_rate * abs(relative_change)
                    )

            adjustments[loss_name] = adjustment

        return adjustments

    def _apply_bounds(self, adjustments: dict[str, float]) -> dict[str, float]:
        """Apply safety bounds to weight adjustments."""
        bounded_adjustments = {}

        for loss_name, adjustment in adjustments.items():
            # Apply bounds
            bounded_adjustment = max(self.min_weight, min(self.max_weight, adjustment))
            bounded_adjustments[loss_name] = bounded_adjustment

            # Update current adjustment with bounds applied
            adjustment_attr = f"current_adjustments_{loss_name}"
            if hasattr(self, adjustment_attr):
                getattr(self, adjustment_attr).fill_(bounded_adjustment)

        return bounded_adjustments

    def _update_adjustment_history(self, adjustments: dict[str, float]) -> None:
        """Update adjustment history for monitoring."""
        for loss_name, adjustment in adjustments.items():
            if loss_name not in self.adaptation_history:
                self.adaptation_history[loss_name] = []

            self.adaptation_history[loss_name].append(adjustment)

            # Keep history to reasonable size
            if len(self.adaptation_history[loss_name]) > 1000:
                self.adaptation_history[loss_name] = self.adaptation_history[loss_name][
                    -1000:
                ]

        # Track significant changes for logging
        current_changes = {}
        for loss_name, adjustment in adjustments.items():
            prev_adjustment = self.last_weight_changes.get(loss_name, 1.0)
            if abs(adjustment - prev_adjustment) > 0.01:  # 1% change
                current_changes[loss_name] = {
                    "previous": prev_adjustment,
                    "current": adjustment,
                    "change": adjustment - prev_adjustment,
                }
                self.adaptation_count += 1

        self.last_weight_changes = adjustments.copy()

        # Log significant changes if monitoring is enabled
        if self.enable_monitoring and current_changes:
            logger = _get_logger()
            changes_str = ", ".join(
                [
                    f"{name}: {data['previous']:.3f} → {data['current']:.3f}"
                    for name, data in current_changes.items()
                ]
            )
            logger.debug(f"Dynamic loss weight adjustments: {changes_str}")

    def get_current_weights(self) -> dict[str, torch.Tensor]:
        """Get current adjusted weights for all tracked losses."""
        current_weights = {}
        for loss_name in self.base_weights.keys():
            base_weight = self.base_weights[loss_name]
            adjustment_attr = f"current_adjustments_{loss_name}"
            adjustment = getattr(self, adjustment_attr, torch.tensor(1.0))
            current_weights[loss_name] = base_weight * adjustment
        return current_weights

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        stats = {
            "iteration": self._iteration.item(),
            "baseline_established": self._baseline_established.item(),
            "adaptation_count": self.adaptation_count,
            "current_weights": {},
            "smoothed_losses": {},
            "loss_velocities": {},
        }

        # Add current state for each loss
        for loss_name in self.base_weights.keys():
            adjustment_attr = f"current_adjustments_{loss_name}"
            smoothed_attr = f"smoothed_losses_{loss_name}"
            velocity_attr = f"loss_velocities_{loss_name}"

            stats["current_weights"][loss_name] = getattr(
                self, adjustment_attr, torch.tensor(1.0)
            ).item()
            stats["smoothed_losses"][loss_name] = getattr(
                self, smoothed_attr, torch.tensor(0.0)
            ).item()
            stats["loss_velocities"][loss_name] = getattr(
                self, velocity_attr, torch.tensor(0.0)
            ).item()

        return stats

    def reset(self, keep_baseline: bool = True) -> None:
        """Reset the scheduler state. Optionally keep the established baseline."""
        # Reset tracking variables
        for loss_name in self.base_weights.keys():
            if not keep_baseline:
                smoothed_attr = f"smoothed_losses_{loss_name}"
                velocity_attr = f"loss_velocities_{loss_name}"
                if hasattr(self, smoothed_attr):
                    getattr(self, smoothed_attr).fill_(0.0)
                if hasattr(self, velocity_attr):
                    getattr(self, velocity_attr).fill_(0.0)
            adjustment_attr = f"current_adjustments_{loss_name}"
            if hasattr(self, adjustment_attr):
                getattr(self, adjustment_attr).fill_(1.0)

        if not keep_baseline:
            self._baseline_established.fill_(False)
            self.adaptation_count = 0
            self.last_weight_changes.clear()

        if self.enable_monitoring:
            logger = _get_logger()
            logger.info(
                f"Dynamic loss scheduler reset (baseline {'kept' if keep_baseline else 'cleared'})"
            )

    def __repr__(self) -> str:
        stats = self.get_monitoring_stats()
        current_weights = stats["current_weights"]

        weight_str = ", ".join(
            [f"{name}: {weight:.3f}" for name, weight in current_weights.items()]
        )

        return (
            f"DynamicLossScheduler("
            f"iteration={stats['iteration']}, "
            f"baseline_established={stats['baseline_established']}, "
            f"adaptation_count={stats['adaptation_count']}, "
            f"current_weights={{{weight_str}}})"
        )


# Utility function to get logger (avoids circular imports)
def _get_logger():
    try:
        from traiNNer.utils import get_root_logger

        return get_root_logger()
    except ImportError:
        # Fallback logger if import fails
        import logging

        return logging.getLogger(__name__)


def create_dynamic_loss_scheduler(
    losses_dict: dict[str, nn.Module], scheduler_config: dict[str, Any]
) -> DynamicLossScheduler:
    """
    Create a DynamicLossScheduler from a dictionary of loss modules.

    Args:
        losses_dict: Dictionary of loss modules keyed by their labels
        scheduler_config: Configuration dictionary for the scheduler

    Returns:
        Configured DynamicLossScheduler instance or None if disabled
    """
    # Check if dynamic loss scheduling is enabled
    enabled = scheduler_config.get("enabled", True)
    if not enabled:
        return None

    # Extract base weights from loss modules
    base_weights = {}
    for loss_label, loss_module in losses_dict.items():
        if hasattr(loss_module, "loss_weight"):
            base_weights[loss_label] = loss_module.loss_weight
        else:
            base_weights[loss_label] = 1.0  # Default weight

    # Validate that all losses have positive weights
    for loss_label, weight in base_weights.items():
        if weight <= 0:
            raise ValueError(f"Loss {loss_label} has non-positive weight: {weight}")

    # Set default configuration values
    default_config = {
        "momentum": 0.9,
        "adaptation_rate": 0.01,
        "min_weight": 1e-6,
        "max_weight": 100.0,
        "adaptation_threshold": 0.1,
        "baseline_iterations": 100,
        "enable_monitoring": True,
    }

    # Merge with provided configuration, excluding the 'enabled' flag
    config = {
        **default_config,
        **{k: v for k, v in scheduler_config.items() if k != "enabled"},
    }

    return DynamicLossScheduler(base_weights=base_weights, **config)
