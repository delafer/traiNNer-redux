#!/usr/bin/env python3
"""
Intelligent Auto-Calibration for Dynamic Loss Scheduling
========================================================

This module provides intelligent auto-calibration of dynamic loss scheduling parameters
to eliminate manual configuration errors and optimize training stability automatically.

Features:
- Architecture-based parameter selection (nano, micro, tiny, xs, s, m, l, xl)
- Dataset complexity analysis and parameter adjustment
- Real-time training stability monitoring
- Automatic parameter correction for training issues
- Zero-configuration user experience

Author: Philip Hofmann
License: MIT
"""

import math
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

import torch
from traiNNer.losses.dynamic_loss_scheduling import DynamicLossScheduler

# Architecture-specific parameter presets optimized for training stability
ARCHITECTURE_PRESETS = {
    "nano": {
        "momentum": 0.85,  # Lower for responsiveness in small models
        "adaptation_rate": 0.015,  # Faster adaptation for rapid learning
        "max_weight": 5.0,  # Lower bounds for stability
        "baseline_iterations": 50,  # Quick baseline establishment
        "adaptation_threshold": 0.04,  # More sensitive to changes
        "min_weight": 1e-6,
    },
    "micro": {
        "momentum": 0.87,
        "adaptation_rate": 0.012,
        "max_weight": 7.5,
        "baseline_iterations": 75,
        "adaptation_threshold": 0.045,
        "min_weight": 1e-6,
    },
    "tiny": {
        "momentum": 0.89,
        "adaptation_rate": 0.010,
        "max_weight": 10.0,
        "baseline_iterations": 100,
        "adaptation_threshold": 0.05,
        "min_weight": 1e-6,
    },
    "xs": {
        "momentum": 0.90,
        "adaptation_rate": 0.009,
        "max_weight": 15.0,
        "baseline_iterations": 125,
        "adaptation_threshold": 0.055,
        "min_weight": 1e-6,
    },
    "s": {
        "momentum": 0.91,
        "adaptation_rate": 0.008,
        "max_weight": 20.0,
        "baseline_iterations": 150,
        "adaptation_threshold": 0.05,
        "min_weight": 1e-6,
    },
    "m": {
        "momentum": 0.92,
        "adaptation_rate": 0.007,
        "max_weight": 30.0,
        "baseline_iterations": 200,
        "adaptation_threshold": 0.05,
        "min_weight": 1e-6,
    },
    "l": {
        "momentum": 0.93,
        "adaptation_rate": 0.006,
        "max_weight": 50.0,
        "baseline_iterations": 250,
        "adaptation_threshold": 0.05,
        "min_weight": 1e-6,
    },
    "xl": {
        "momentum": 0.94,
        "adaptation_rate": 0.005,
        "max_weight": 75.0,
        "baseline_iterations": 300,
        "adaptation_threshold": 0.05,
        "min_weight": 1e-6,
    },
}


def detect_model_architecture(model) -> str:
    """
    Automatically detect the ParagonSR2 architecture variant.

    Args:
        model: PyTorch model instance

    Returns:
        Architecture variant string (nano, micro, tiny, xs, s, m, l, xl)
    """
    # Check for known architecture patterns
    model_name = model.__class__.__name__.lower()

    if "nano" in model_name:
        return "nano"
    elif "micro" in model_name:
        return "micro"
    elif "tiny" in model_name:
        return "tiny"
    elif "xs" in model_name:
        return "xs"
    elif "paragonsr2" in model_name:
        # For base ParagonSR2 class, try to infer from model parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate based on parameter count (rough approximation)
        if num_params < 50000:  # ~0.02M params
            return "nano"
        elif num_params < 100000:  # ~0.04M params
            return "micro"
        elif num_params < 200000:  # ~0.08M params
            return "tiny"
        elif num_params < 300000:  # ~0.12M params
            return "xs"
        elif num_params < 500000:  # ~0.28M params
            return "s"
        elif num_params < 1000000:  # ~0.65M params
            return "m"
        elif num_params < 2000000:  # ~1.8M params
            return "l"
        else:  # ~3.8M+ params
            return "xl"

    # Default fallback
    return "s"


def analyze_dataset_complexity(dataloader) -> float:
    """
    Analyze dataset complexity from a sample batch.

    Args:
        dataloader: PyTorch DataLoader instance

    Returns:
        Complexity score between 0.0 (simple) and 1.0 (complex)
    """
    try:
        # Get first batch for analysis
        sample_batch = next(iter(dataloader))

        if "lq" in sample_batch and "gt" in sample_batch:
            gt = sample_batch["gt"]  # Ground truth images

            # Calculate complexity metrics
            complexity_score = 0.0

            # 1. Image variance (high variance = complex textures)
            gt_var = torch.var(gt)
            complexity_score += min(gt_var * 5, 0.3)  # Weight: 30%

            # 2. High frequency content (edges, textures)
            # Calculate gradients to detect edges and textures
            if gt.dim() == 4:  # (B, C, H, W)
                grad_x = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
                grad_y = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
                edge_density = (grad_x.mean() + grad_y.mean()) * 20
                complexity_score += min(edge_density, 0.4)  # Weight: 40%

            # 3. Color variation and contrast
            color_range = gt.max() - gt.min()
            complexity_score += min(color_range * 3, 0.3)  # Weight: 30%

            # Normalize to 0-1 range
            complexity_score = min(complexity_score, 1.0)

            return complexity_score

    except Exception as e:
        print(f"Warning: Dataset complexity analysis failed: {e}")
        return 0.5  # Default to moderate complexity

    return 0.5  # Fallback


def adjust_parameters_for_dataset(
    base_params: dict[str, float], complexity_score: float
) -> dict[str, float]:
    """
    Adjust parameters based on dataset complexity.

    Args:
        base_params: Base parameters from architecture preset
        complexity_score: Dataset complexity score (0.0-1.0)

    Returns:
        Adjusted parameters
    """
    adjusted_params = base_params.copy()

    if complexity_score < 0.3:  # Simple dataset
        # Simple datasets benefit from more responsive parameters
        adjusted_params["momentum"] *= 0.95  # Lower momentum for responsiveness
        adjusted_params["adaptation_rate"] *= 1.2  # Faster adaptation
        adjusted_params["max_weight"] *= 0.8  # Lower bounds for stability
        adjusted_params["adaptation_threshold"] *= 0.9  # More sensitive

    elif complexity_score > 0.7:  # Complex dataset
        # Complex datasets need more stable parameters
        adjusted_params["momentum"] *= 1.05  # Higher momentum for stability
        adjusted_params["adaptation_rate"] *= 0.8  # Slower, more careful adaptation
        adjusted_params["max_weight"] *= 1.3  # Higher bounds for flexibility
        adjusted_params["adaptation_threshold"] *= 1.1  # Less sensitive

    # Moderate complexity: use base parameters

    return adjusted_params


def get_training_phase_adjusted_parameters(
    base_params: dict[str, float], current_iter: int, total_iter: int
) -> dict[str, float]:
    """
    Adjust parameters based on training phase (early/mid/late training).

    Args:
        base_params: Base parameters
        current_iter: Current training iteration
        total_iter: Total training iterations

    Returns:
        Phase-adjusted parameters
    """
    progress = current_iter / total_iter

    if progress < 0.1:  # First 10% - Early training (stability focus)
        return {
            "momentum": base_params["momentum"] * 0.9,  # More conservative
            "adaptation_rate": base_params["adaptation_rate"] * 0.7,  # Slower
            "max_weight": base_params["max_weight"] * 0.5,  # Lower bounds
            "adaptation_threshold": base_params["adaptation_threshold"]
            * 0.8,  # More sensitive
            "baseline_iterations": base_params["baseline_iterations"],
        }
    elif progress < 0.8:  # Middle training (10%-80%) - Standard parameters
        return base_params
    else:  # Late training (80%+) - Fine-tuning focus
        return {
            "momentum": base_params["momentum"] * 1.02,  # Slightly more aggressive
            "adaptation_rate": base_params["adaptation_rate"]
            * 1.1,  # Faster for fine-tuning
            "max_weight": base_params["max_weight"] * 1.2,  # Higher bounds
            "adaptation_threshold": base_params["adaptation_threshold"]
            * 1.05,  # Less sensitive
            "baseline_iterations": base_params["baseline_iterations"],
        }


class IntelligentDynamicLossScheduler(DynamicLossScheduler):
    """
    Enhanced DynamicLossScheduler with intelligent auto-calibration.

    This version automatically:
    - Detects model architecture and selects optimal parameters
    - Analyzes dataset complexity and adjusts parameters accordingly
    - Monitors training stability and auto-corrects problematic configurations
    - Provides zero-configuration user experience
    """

    def __init__(
        self,
        base_weights: dict[str, float],
        model: torch.nn.Module | None = None,
        dataloader: torch.utils.data.DataLoader | None = None,
        total_iter: int | None = None,
        auto_calibrate: bool = True,
        enable_monitoring: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize intelligent auto-calibrating dynamic loss scheduler.

        Args:
            base_weights: Dictionary of base loss weights for each loss type
            model: PyTorch model for architecture detection
            dataloader: DataLoader for dataset complexity analysis
            total_iter: Total training iterations for phase adjustment
            auto_calibrate: Enable intelligent auto-calibration
            enable_monitoring: Enable detailed monitoring and logging
            **kwargs: Additional parameters (ignored if auto_calibrate=True)
        """
        super().__init__(
            base_weights=base_weights,
            momentum=0.9,  # Temporary, will be overridden
            adaptation_rate=0.01,  # Temporary, will be overridden
            min_weight=1e-6,
            max_weight=100.0,  # Temporary, will be overridden
            adaptation_threshold=0.05,
            baseline_iterations=100,
            enable_monitoring=enable_monitoring,
        )

        self.auto_calibrate = auto_calibrate
        self.model = model
        self.dataloader = dataloader
        self.total_iter = total_iter
        self.current_iter = 0

        # Auto-calculate optimal parameters
        if auto_calibrate:
            self._auto_calibrate_parameters()

        # Training stability monitoring
        self.stability_window = 100  # Monitor last 100 iterations
        self.loss_history = deque(maxlen=self.stability_window)
        self.adjustment_history = []
        self.correction_count = 0

        # Initialize with calculated parameters
        if auto_calibrate:
            self._apply_calculated_parameters()

    def _auto_calibrate_parameters(self) -> None:
        """Auto-calculate optimal parameters based on model and dataset."""
        logger = _get_logger()

        # Step 1: Detect model architecture
        if self.model is not None:
            arch_variant = detect_model_architecture(self.model)
            base_params = ARCHITECTURE_PRESETS.get(
                arch_variant, ARCHITECTURE_PRESETS["s"]
            )
            logger.info(f"Auto-detected model architecture: {arch_variant}")
        else:
            # Fallback to medium size if model not provided
            base_params = ARCHITECTURE_PRESETS["s"]
            logger.info("Model not provided, using default parameters for 's' variant")

        # Step 2: Analyze dataset complexity
        if self.dataloader is not None:
            complexity_score = analyze_dataset_complexity(self.dataloader)
            adjusted_params = adjust_parameters_for_dataset(
                base_params, complexity_score
            )
            logger.info(
                f"Dataset complexity: {complexity_score:.3f} (adjusted parameters)"
            )
        else:
            adjusted_params = base_params
            logger.info("Dataset not provided, using base architecture parameters")

        # Store calculated parameters
        self.calculated_params = adjusted_params

        # Log parameter selection
        logger.info(
            f"Auto-calibrated dynamic loss scheduling parameters: "
            f"momentum={adjusted_params['momentum']:.3f}, "
            f"adaptation_rate={adjusted_params['adaptation_rate']:.3f}, "
            f"max_weight={adjusted_params['max_weight']:.1f}, "
            f"baseline_iterations={adjusted_params['baseline_iterations']}"
        )

    def _apply_calculated_parameters(self) -> None:
        """Apply the auto-calculated parameters to the scheduler."""
        params = self.calculated_params
        self.momentum = params["momentum"]
        self.adaptation_rate = params["adaptation_rate"]
        self.min_weight = params["min_weight"]
        self.max_weight = params["max_weight"]
        self.adaptation_threshold = params["adaptation_threshold"]
        self.baseline_iterations = params["baseline_iterations"]

        # Update base weights if needed
        for loss_name in self.base_weights.keys():
            adjustment_attr = f"current_adjustments_{loss_name}"
            if hasattr(self, adjustment_attr):
                getattr(self, adjustment_attr).fill_(1.0)

    def forward(
        self,
        current_losses: dict[str, float | torch.Tensor | dict[str, torch.Tensor]],
        current_iter: int,
    ) -> dict[str, float]:
        """
        Enhanced forward pass with stability monitoring and auto-correction.

        Args:
            current_losses: Dictionary of current loss values for each loss type
            current_iter: Current training iteration

        Returns:
            Dictionary of adjusted weight multipliers for each loss type
        """
        self.current_iter = current_iter

        # Apply training phase adjustments if total_iter is known
        if self.total_iter is not None:
            phase_adjusted_params = get_training_phase_adjusted_parameters(
                self.calculated_params, current_iter, self.total_iter
            )

            # Update current parameters based on training phase
            self.momentum = phase_adjusted_params["momentum"]
            self.adaptation_rate = phase_adjusted_params["adaptation_rate"]
            self.max_weight = phase_adjusted_params["max_weight"]
            self.adaptation_threshold = phase_adjusted_params["adaptation_threshold"]

        # Monitor training stability
        if self.auto_calibrate and current_iter > 500:  # Start monitoring after warmup
            self._monitor_stability(current_losses, current_iter)

        # Call parent class for standard dynamic loss scheduling
        adjusted_weights = super().forward(current_losses, current_iter)

        return adjusted_weights

    def _monitor_stability(self, current_losses: dict, current_iter: int) -> None:
        """Monitor training stability and auto-correct if issues detected."""

        # Calculate total loss for stability tracking
        if isinstance(current_losses, dict):
            total_loss = sum(
                abs(v) if isinstance(v, (int, float)) else v.item()
                for v in current_losses.values()
            )
        else:
            total_loss = abs(current_losses)

        self.loss_history.append((current_iter, total_loss))

        # Check for training instability if we have enough history
        if len(self.loss_history) >= 20:
            stability_issue = self._detect_training_instability()

            if stability_issue:
                self._auto_correct_parameters(stability_issue, current_iter)

    def _detect_training_instability(self) -> str | None:
        """Detect if training is becoming unstable."""

        recent_losses = [loss for _, loss in list(self.loss_history)[-20:]]

        if len(recent_losses) < 10:
            return None

        early_avg = sum(recent_losses[:5]) / 5
        late_avg = sum(recent_losses[-5:]) / 5

        # Check for exponential loss growth (training degradation)
        if late_avg > early_avg * 2.5:  # Loss increased by 150%
            return "exponential_loss_growth"

        # Check for high loss variance (unstable training)
        variance = sum((loss - late_avg) ** 2 for loss in recent_losses) / len(
            recent_losses
        )
        cv = (variance**0.5) / late_avg if late_avg > 0 else 0

        if cv > 0.8:  # High coefficient of variation
            return "high_loss_variance"

        return None

    def _auto_correct_parameters(self, stability_issue: str, current_iter: int) -> None:
        """Automatically correct parameters when training instability is detected."""

        logger = _get_logger()
        original_params = {
            "momentum": self.momentum,
            "adaptation_rate": self.adaptation_rate,
            "max_weight": self.max_weight,
        }

        if stability_issue == "exponential_loss_growth":
            # Make parameters more conservative to prevent degradation
            self.momentum = min(0.99, self.momentum * 1.08)  # Higher momentum
            self.adaptation_rate = max(
                0.001, self.adaptation_rate * 0.7
            )  # Slower adaptation
            self.max_weight = max(1.0, self.max_weight * 0.6)  # Lower bounds

        elif stability_issue == "high_loss_variance":
            # Increase stability parameters
            self.momentum = min(0.99, self.momentum * 1.05)
            self.adaptation_rate = max(0.001, self.adaptation_rate * 0.85)
            self.max_weight = max(1.0, self.max_weight * 0.7)

        # Record the correction
        correction = {
            "iteration": current_iter,
            "issue": stability_issue,
            "original": original_params,
            "corrected": {
                "momentum": self.momentum,
                "adaptation_rate": self.adaptation_rate,
                "max_weight": self.max_weight,
            },
        }

        self.adjustment_history.append(correction)
        self.correction_count += 1

        # Log the correction
        logger.warning(
            f"Auto-corrected dynamic loss scheduling due to {stability_issue} at iteration {current_iter}"
        )


def create_intelligent_dynamic_loss_scheduler(
    losses_dict: dict[str, torch.nn.Module],
    model: torch.nn.Module | None = None,
    dataloader: torch.utils.data.DataLoader | None = None,
    total_iter: int | None = None,
    scheduler_config: dict[str, Any] | None = None,
) -> IntelligentDynamicLossScheduler:
    """
    Create an intelligent auto-calibrating dynamic loss scheduler.

    This function provides a simplified interface for creating the intelligent scheduler
    with automatic parameter optimization.

    Args:
        losses_dict: Dictionary of loss modules keyed by their labels
        model: PyTorch model for architecture detection
        dataloader: DataLoader for dataset complexity analysis
        total_iter: Total training iterations for phase adjustment
        scheduler_config: Configuration dictionary (auto_calibrate takes precedence)

    Returns:
        IntelligentDynamicLossScheduler instance with optimal parameters
    """
    if scheduler_config is None:
        scheduler_config = {}

    # Check if auto_calibrate is explicitly set
    auto_calibrate = scheduler_config.get("auto_calibrate", True)

    # Extract base weights from loss modules
    base_weights = {}
    for loss_label, loss_module in losses_dict.items():
        if hasattr(loss_module, "loss_weight"):
            base_weights[loss_label] = loss_module.loss_weight
        else:
            base_weights[loss_label] = 1.0

    # Validate that all losses have positive weights
    for loss_label, weight in base_weights.items():
        if weight <= 0:
            raise ValueError(f"Loss {loss_label} has non-positive weight: {weight}")

    # Create intelligent scheduler with auto-calibration
    return IntelligentDynamicLossScheduler(
        base_weights=base_weights,
        model=model,
        dataloader=dataloader,
        total_iter=total_iter,
        auto_calibrate=auto_calibrate,
        enable_monitoring=scheduler_config.get("enable_monitoring", True),
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


if __name__ == "__main__":
    # Example usage and testing
    import torch
    from torch import nn

    # Create a simple test model (simulating ParagonSR2 Nano)
    class TestModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 12, 3, 1, 1)  # ~12M params equivalent

    # Test the intelligent auto-calibration
    model = TestModel()

    # Create mock losses
    losses = {"l_g_l1": nn.Module(), "l_g_ssim": nn.Module()}
    losses["l_g_l1"].loss_weight = 1.0
    losses["l_g_ssim"].loss_weight = 0.05

    # Create intelligent scheduler
    scheduler = create_intelligent_dynamic_loss_scheduler(
        losses_dict=losses, model=model, total_iter=40000
    )

    # Test functionality
    print("Intelligent Dynamic Loss Scheduler Test")
    print(f"Auto-calibrated parameters: {scheduler.calculated_params}")
    print(f"Model architecture detected: {detect_model_architecture(model)}")

    # Simulate training iterations
    for i in range(1000, 5000, 1000):
        test_losses = {"l_g_l1": 0.03, "l_g_ssim": 0.015}
        adjusted_weights = scheduler(test_losses, i)
        print(f"Iteration {i}: {adjusted_weights}")
