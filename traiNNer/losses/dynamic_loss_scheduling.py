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

        # Convert parameters to proper types and validate
        try:
            momentum = float(momentum)
            adaptation_rate = float(adaptation_rate)
            min_weight = float(min_weight)
            max_weight = float(max_weight)
            adaptation_threshold = float(adaptation_threshold)
            baseline_iterations = int(baseline_iterations)
            enable_monitoring = bool(enable_monitoring)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert scheduler parameters to proper types: {e}"
            )

        # Validate parameter ranges
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
                scalar_losses[loss_name] = float(loss_value.mean().abs().detach())
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

    # Check for auto-calibration mode
    auto_calibrate = scheduler_config.get("auto_calibrate", False)

    if auto_calibrate:
        # Use intelligent auto-calibration system
        return _create_intelligent_loss_scheduler(losses_dict, scheduler_config)
    else:
        # Use traditional manual configuration
        return _create_manual_loss_scheduler(losses_dict, scheduler_config)


def _create_intelligent_loss_scheduler(
    losses_dict: dict[str, nn.Module], scheduler_config: dict[str, Any]
) -> DynamicLossScheduler:
    """
    Create an intelligent dynamic loss scheduler with automatic calibration.

    Args:
        losses_dict: Dictionary of loss modules keyed by their labels
        scheduler_config: Configuration dictionary for the scheduler

    Returns:
        Configured DynamicLossScheduler with intelligent auto-calibration
    """
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

    # Get architecture type from network configuration (if available)
    architecture_type = scheduler_config.get("architecture_type", "unknown")

    # Get training configuration for intelligent analysis
    training_config = scheduler_config.get("training_config", {})
    total_iterations = training_config.get("total_iterations", 40000)
    dataset_info = training_config.get("dataset_info", {})

    # Determine optimal parameters using intelligent presets
    intelligent_params = _determine_intelligent_parameters(
        architecture_type=architecture_type,
        total_iterations=total_iterations,
        dataset_info=dataset_info,
        loss_names=list(base_weights.keys()),
        scheduler_config=scheduler_config,
    )

    # Enable comprehensive monitoring for intelligent system
    intelligent_params["enable_monitoring"] = True

    # Log the intelligent calibration
    logger = _get_logger()
    logger.info(
        f"🎯 Intelligent Auto-Calibration: {architecture_type} architecture detected",
        extra={"markup": True},
    )
    logger.info(
        f"🧠 Auto-calibrated parameters: {intelligent_params}", extra={"markup": True}
    )
    logger.info(
        "✅ Dynamic loss scheduling ready - user only needs to set auto_calibrate: true",
        extra={"markup": True},
    )

    # Ensure all parameters are properly typed before passing to DynamicLossScheduler
    typed_params = {}
    for key, value in intelligent_params.items():
        if key == "momentum":
            typed_params[key] = float(value)
        elif key == "adaptation_rate":
            typed_params[key] = float(value)
        elif key == "min_weight":
            typed_params[key] = float(value)
        elif key == "max_weight":
            typed_params[key] = float(value)
        elif key == "adaptation_threshold":
            typed_params[key] = float(value)
        elif key == "baseline_iterations":
            typed_params[key] = int(value)
        elif key == "enable_monitoring":
            if isinstance(value, str):
                typed_params[key] = value.lower() in ("true", "1", "yes", "on")
            else:
                typed_params[key] = bool(value)
        else:
            typed_params[key] = value

    return DynamicLossScheduler(base_weights=base_weights, **typed_params)


def _determine_intelligent_parameters(
    architecture_type: str,
    total_iterations: int,
    dataset_info: dict,
    loss_names: list,
    scheduler_config: dict,
) -> dict[str, Any]:
    """
    Intelligently determine optimal parameters based on architecture and context.

    Args:
        architecture_type: Type of neural network architecture
        total_iterations: Total training iterations planned
        dataset_info: Information about the training dataset
        loss_names: Names of loss functions being used
        scheduler_config: Original configuration (may contain overrides)

    Returns:
        Dictionary of optimal parameters for the scheduler
    """

    # Architecture-specific parameter presets
    ARCHITECTURE_PRESETS = {
        # ParagonSR2 variants - optimized based on training analysis
        "paragonsr2_nano": {
            "momentum": 0.85,
            "adaptation_rate": 0.015,
            "min_weight": 1e-6,
            "max_weight": 5.0,
            "adaptation_threshold": 0.04,
            "baseline_iterations": 50,
        },
        "paragonsr2_micro": {
            "momentum": 0.87,
            "adaptation_rate": 0.012,
            "min_weight": 1e-6,
            "max_weight": 7.5,
            "adaptation_threshold": 0.05,
            "baseline_iterations": 75,
        },
        "paragonsr2_tiny": {
            "momentum": 0.89,
            "adaptation_rate": 0.010,
            "min_weight": 1e-6,
            "max_weight": 10.0,
            "adaptation_threshold": 0.06,
            "baseline_iterations": 100,
        },
        "paragonsr2_xs": {
            "momentum": 0.91,
            "adaptation_rate": 0.008,
            "min_weight": 1e-6,
            "max_weight": 15.0,
            "adaptation_threshold": 0.07,
            "baseline_iterations": 125,
        },
        "paragonsr2_s": {
            "momentum": 0.93,
            "adaptation_rate": 0.006,
            "min_weight": 1e-6,
            "max_weight": 20.0,
            "adaptation_threshold": 0.08,
            "baseline_iterations": 150,
        },
        "paragonsr2_m": {
            "momentum": 0.95,
            "adaptation_rate": 0.005,
            "min_weight": 1e-6,
            "max_weight": 30.0,
            "adaptation_threshold": 0.10,
            "baseline_iterations": 200,
        },
        "paragonsr2_l": {
            "momentum": 0.96,
            "adaptation_rate": 0.004,
            "min_weight": 1e-6,
            "max_weight": 50.0,
            "adaptation_threshold": 0.12,
            "baseline_iterations": 250,
        },
        "paragonsr2_xl": {
            "momentum": 0.97,
            "adaptation_rate": 0.003,
            "min_weight": 1e-6,
            "max_weight": 100.0,
            "adaptation_threshold": 0.15,
            "baseline_iterations": 300,
        },
    }

    # Normalize architecture type for matching
    arch_key = architecture_type.lower()
    if "paragonsr2" in arch_key:
        # Extract specific variant (nano, micro, tiny, etc.) - must be complete word boundaries
        import re

        for variant in ["nano", "micro", "tiny", "xs", "s", "m", "l", "xl"]:
            # Use regex to match variant as complete word/separator
            if re.search(r"\b" + re.escape(variant) + r"\b", arch_key):
                preset_key = f"paragonsr2_{variant}"
                break
        else:
            preset_key = "paragonsr2_nano"  # Default fallback
    # Generic fallback presets for unknown architectures
    elif "nano" in arch_key or "small" in arch_key:
        preset_key = "paragonsr2_nano"
    elif "micro" in arch_key:
        preset_key = "paragonsr2_micro"
    elif "tiny" in arch_key:
        preset_key = "paragonsr2_tiny"
    elif "xs" in arch_key:
        preset_key = "paragonsr2_xs"
    elif "small" in arch_key or (
        arch_key.endswith("_s") and len(arch_key.split("_")) > 1
    ):
        preset_key = "paragonsr2_s"
    elif "medium" in arch_key or (
        arch_key.endswith("_m") and len(arch_key.split("_")) > 1
    ):
        preset_key = "paragonsr2_m"
    elif "large" in arch_key or (
        arch_key.endswith("_l") and len(arch_key.split("_")) > 1
    ):
        preset_key = "paragonsr2_l"
    elif "xl" in arch_key or "extra" in arch_key:
        preset_key = "paragonsr2_xl"
    else:
        preset_key = "paragonsr2_nano"  # Conservative fallback

    # Get base preset
    if preset_key in ARCHITECTURE_PRESETS:
        params = ARCHITECTURE_PRESETS[preset_key].copy()
    else:
        # Conservative defaults for unknown architectures
        params = ARCHITECTURE_PRESETS["paragonsr2_nano"].copy()

    # Training phase adjustments
    if total_iterations < 10000:
        # Short training - more aggressive adaptation
        params["adaptation_rate"] *= 1.5
        params["baseline_iterations"] = max(25, params["baseline_iterations"] // 2)
    elif total_iterations > 50000:
        # Long training - more conservative, stable adaptation
        params["adaptation_rate"] *= 0.7
        params["baseline_iterations"] = min(
            400, int(params["baseline_iterations"] * 1.5)
        )

    # Dataset complexity adjustments (if provided)
    if dataset_info:
        texture_variance = dataset_info.get("texture_variance", 0.5)
        edge_density = dataset_info.get("edge_density", 0.5)
        color_variation = dataset_info.get("color_variation", 0.5)
        overall_complexity = dataset_info.get("overall_complexity", 0.5)

        # Use the overall complexity score if available, otherwise compute it
        complexity_score = overall_complexity

        # High complexity datasets need more conservative adaptation
        if complexity_score > 0.7:
            # Complex datasets: more conservative to avoid instability
            params["momentum"] *= 0.9  # More responsive
            params["adaptation_rate"] *= 1.2  # Faster adaptation
            params["adaptation_threshold"] *= 1.5  # Less sensitive to noise
            params["max_weight"] *= 0.8  # Lower ceiling for stability
        elif complexity_score < 0.3:
            # Simple datasets: can be more aggressive
            params["momentum"] *= 1.1  # More stable
            params["adaptation_rate"] *= 0.8  # Slower adaptation
            params["adaptation_threshold"] *= 0.7  # More sensitive
            params["max_weight"] *= 1.2  # Higher ceiling for exploration

        # Texture-specific adjustments
        if texture_variance > 0.6:
            # High texture variance: more sensitive to texture details
            params["adaptation_rate"] *= 1.1
            params["momentum"] *= 0.95
        elif texture_variance < 0.4:
            # Low texture variance: less sensitive
            params["adaptation_rate"] *= 0.9
            params["momentum"] *= 1.05

        # Edge-specific adjustments
        if edge_density > 0.6:
            # High edge density: lots of details, need careful handling
            params["adaptation_threshold"] *= 1.2
            params["momentum"] *= 0.92
        elif edge_density < 0.4:
            # Low edge density: smoother images, can adapt faster
            params["adaptation_threshold"] *= 0.8
            params["momentum"] *= 1.08

        # Color-specific adjustments
        if color_variation > 0.6:
            # High color variation: diverse lighting/colors
            params["adaptation_rate"] *= 1.05
            params["max_weight"] *= 0.9
        elif color_variation < 0.4:
            # Low color variation: consistent colors
            params["adaptation_rate"] *= 0.95
            params["max_weight"] *= 1.1

    # Loss type adjustments
    if "gan" in "".join(loss_names).lower():
        # GAN training needs more careful handling
        params["adaptation_threshold"] *= 1.5  # Less sensitive to noise
        params["max_weight"] *= 0.8  # Lower ceiling for GAN losses

    # Apply any user-provided overrides (except auto_calibrate itself)
    for key, value in scheduler_config.items():
        if key not in [
            "enabled",
            "auto_calibrate",
            "architecture_type",
            "training_config",
            "dataset_info",
        ]:
            params[key] = value

    return params


def _create_manual_loss_scheduler(
    losses_dict: dict[str, nn.Module], scheduler_config: dict[str, Any]
) -> DynamicLossScheduler:
    """
    Create a traditional manual dynamic loss scheduler.

    Args:
        losses_dict: Dictionary of loss modules keyed by their labels
        scheduler_config: Configuration dictionary for the scheduler

    Returns:
        Configured DynamicLossScheduler with manual parameters
    """
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

    # Convert string values to proper types
    type_conversions = {
        "momentum": float,
        "adaptation_rate": float,
        "min_weight": float,
        "max_weight": float,
        "adaptation_threshold": float,
        "baseline_iterations": int,
        "enable_monitoring": bool,
    }

    for key, converter in type_conversions.items():
        if key in config:
            try:
                if key == "enable_monitoring":
                    # Handle boolean conversion
                    if isinstance(config[key], str):
                        config[key] = config[key].lower() in ("true", "1", "yes", "on")
                    else:
                        config[key] = bool(config[key])
                elif key == "baseline_iterations":
                    # Handle int conversion - first convert to float, then to int
                    config[key] = int(float(config[key]))
                else:
                    config[key] = converter(config[key])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Failed to convert {key}={config[key]} to {converter.__name__}: {e}"
                )

    return DynamicLossScheduler(base_weights=base_weights, **config)
