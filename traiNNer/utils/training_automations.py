"""
Training Automations Module

Provides intelligent automation for key training parameters to optimize training
performance, stability, and efficiency. Phase 1 automations include:

1. Intelligent Learning Rate Scheduling
2. Dynamic Batch Size Optimization
3. Adaptive Gradient Clipping
4. Early Stopping

All automations include comprehensive safety measures and backward compatibility.
"""

import logging
import math
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.optim.lr_scheduler import LRScheduler

from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.registry import Registry

logger = logging.getLogger(__name__)

AUTOMATION_REGISTRY = Registry("automation")


class TrainingAutomationBase:
    """Base class for all training automations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.enabled = config.get("enabled", False)
        self.name = self.__class__.__name__
        self.iteration = 0
        self.enabled_iterations = 0

        # Safety measures
        self.fallback_config = config.get("fallback", {})
        self.max_adjustments = config.get("max_adjustments", 100)
        self.adjustment_count = 0
        self.adjustment_history = []

        if self.enabled:
            logger.info(f"Automation {self.name} enabled with config: {config}")

    def update_iteration(self, iteration: int) -> None:
        """Update current iteration."""
        self.iteration = iteration
        if self.enabled:
            self.enabled_iterations += 1

    def should_disable(self) -> bool:
        """Check if automation should be disabled due to issues."""
        if self.adjustment_count >= self.max_adjustments:
            logger.warning(
                f"Automation {self.name} disabled due to too many adjustments ({self.adjustment_count})"
            )
            return True
        return False

    def record_adjustment(
        self, parameter: str, old_value: Any, new_value: Any, reason: str
    ) -> None:
        """Record an automation adjustment for logging and safety."""
        self.adjustment_count += 1
        adjustment_info = {
            "iteration": self.iteration,
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }
        self.adjustment_history.append(adjustment_info)

        logger.info(
            f"Automation {self.name}: {parameter} adjusted from {old_value} to {new_value} "
            f"at iteration {self.iteration} due to: {reason}"
        )

    def get_fallback_value(self, parameter: str, current_value: Any) -> Any:
        """Get fallback value for a parameter."""
        return self.fallback_config.get(parameter, current_value)

    def safe_adjust(
        self,
        parameter: str,
        current_value: Any,
        new_value: Any,
        reason: str,
        validate_func: Union[callable, None] = None,
    ) -> Any:
        """Safely adjust a parameter with validation and fallback."""
        if not self.enabled:
            return current_value

        if validate_func and not validate_func(new_value):
            logger.warning(
                f"Automation {self.name}: Invalid value {new_value} for {parameter}, keeping {current_value}"
            )
            return current_value

        if abs(float(new_value) - float(current_value)) < 1e-6:
            return current_value  # No significant change

        if self.should_disable():
            return current_value

        self.record_adjustment(parameter, current_value, new_value, reason)
        return new_value


@AUTOMATION_REGISTRY.register()
class IntelligentLearningRateScheduler(TrainingAutomationBase):
    """
    Intelligent Learning Rate Scheduler

    Monitors training progress and automatically adjusts learning rate scheduling
    based on loss curves, convergence patterns, and training stability.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Configuration
        self.monitor_loss = config.get("monitor_loss", True)
        self.monitor_validation = config.get("monitor_validation", True)
        self.adaptation_threshold = config.get("adaptation_threshold", 0.02)
        self.plateau_patience = config.get("plateau_patience", 1000)
        self.improvement_threshold = config.get("improvement_threshold", 0.001)
        self.min_lr_factor = config.get("min_lr_factor", 0.1)
        self.max_lr_factor = config.get("max_lr_factor", 2.0)

        # State tracking
        self.loss_history = deque(maxlen=200)
        self.val_metric_history = deque(maxlen=50)
        self.lr_adjustment_history = deque(maxlen=10)
        self.baseline_loss = None
        self.best_loss = float("inf")
        self.best_loss_iteration = 0
        self.plateau_counter = 0
        self.lr_multipliers = {}  # Per parameter group

        # Architecture-specific optimizations
        self.architecture_hints = config.get("architecture_hints", {})

    def update_loss_tracking(self, loss_value: float) -> None:
        """Update loss tracking for intelligent scheduling."""
        if not self.enabled:
            return

        self.loss_history.append(loss_value)

        # Establish baseline after 100 iterations
        if self.baseline_loss is None and len(self.loss_history) >= 100:
            self.baseline_loss = sum(list(self.loss_history)[-100:]) / 100
            logger.info(
                f"Automation {self.name}: Baseline loss established: {self.baseline_loss:.6f}"
            )

        # Track best loss for plateau detection
        if loss_value < self.best_loss - self.improvement_threshold:
            self.best_loss = loss_value
            self.best_loss_iteration = self.iteration
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        # Check if we should adjust learning rate
        if self._should_adjust_learning_rate():
            suggested_lr_multiplier = self._calculate_lr_multiplier()
            if suggested_lr_multiplier != 1.0:
                self._apply_lr_multiplier(suggested_lr_multiplier)

    def update_validation_tracking(self, metrics: dict[str, float]) -> None:
        """Update validation metric tracking."""
        if not self.enabled or not self.monitor_validation:
            return

        # Extract primary metric (prefer PSNR, then SSIM, then first metric)
        primary_metric = None
        for metric_name in ["psnr", "ssim", "lpips", "fid"]:
            if metric_name in metrics:
                primary_metric = metrics[metric_name]
                if metric_name in ["psnr", "ssim"]:
                    primary_metric = metrics[metric_name]  # Higher is better
                else:
                    primary_metric = -metrics[metric_name]  # Lower is better
                break

        if primary_metric is not None:
            self.val_metric_history.append(primary_metric)

            # Check for validation-based LR adjustments
            if len(self.val_metric_history) >= 10:
                recent_metrics = list(self.val_metric_history)[-10:]
                if self._detect_validation_plateau(recent_metrics):
                    logger.info(
                        f"Automation {self.name}: Validation plateau detected, reducing learning rate"
                    )
                    self._apply_lr_multiplier(0.8)  # Conservative reduction

    def _should_adjust_learning_rate(self) -> bool:
        """Determine if learning rate should be adjusted."""
        if len(self.loss_history) < 200:
            return False

        # Check for plateau (no improvement for plateau_patience iterations)
        if self.plateau_counter >= self.plateau_patience:
            return True

        # Check for loss divergence (loss increasing significantly)
        recent_losses = list(self.loss_history)[-50:]
        if len(recent_losses) >= 50:
            early_loss = sum(recent_losses[:25]) / 25
            late_loss = sum(recent_losses[-25:]) / 25

            if late_loss > early_loss * 1.05:  # 5% increase
                return True

        return False

    def _calculate_lr_multiplier(self) -> float:
        """Calculate suggested learning rate multiplier."""
        if self.plateau_counter >= self.plateau_patience:
            # Reduce LR when stuck in plateau
            return 0.8
        elif self.plateau_counter >= self.plateau_patience // 2:
            # Slight reduction for early plateau detection
            return 0.9
        elif self.baseline_loss and self.loss_history[-1] < self.baseline_loss * 0.95:
            # Increase LR if loss improved significantly
            return 1.1

        return 1.0

    def _apply_lr_multiplier(self, multiplier: float) -> None:
        """Apply learning rate multiplier."""
        if multiplier == 1.0:
            return

        # Clamp multiplier within bounds
        multiplier = max(self.min_lr_factor, min(self.max_lr_factor, multiplier))

        # Record adjustment
        reason = (
            f"plateau detection ({self.plateau_counter} iterations)"
            if self.plateau_counter >= self.plateau_patience
            else "loss divergence"
        )
        self.lr_adjustment_history.append(multiplier)

        logger.info(
            f"Automation {self.name}: Suggested LR multiplier {multiplier:.2f} for reason: {reason}"
        )

    def _detect_validation_plateau(self, recent_metrics: list[float]) -> bool:
        """Detect if validation metrics have plateaued."""
        if len(recent_metrics) < 10:
            return False

        # Check for no improvement over recent iterations
        best_metric = max(recent_metrics)
        current_metric = recent_metrics[-1]

        return (
            current_metric < best_metric - self.improvement_threshold
            and len(
                [
                    m
                    for m in recent_metrics[-5:]
                    if m > current_metric + self.improvement_threshold
                ]
            )
            == 0
        )


@AUTOMATION_REGISTRY.register()
class DynamicBatchSizeOptimizer(TrainingAutomationBase):
    """
    Dynamic Batch Size Optimizer

    Monitors VRAM usage and automatically adjusts batch size to optimize
    training efficiency while preventing out-of-memory errors.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Configuration
        self.target_vram_usage = config.get("target_vram_usage", 0.85)
        self.safety_margin = config.get("safety_margin", 0.05)
        self.adjustment_frequency = config.get("adjustment_frequency", 100)
        self.min_batch_size = config.get("min_batch_size", 1)
        self.max_batch_size = config.get("max_batch_size", 32)
        self.vram_history_size = config.get("vram_history_size", 50)

        # State tracking
        self.vram_history = deque(maxlen=self.vram_history_size)
        self.current_batch_size = None
        self.target_batch_size = None
        self.adjustment_cooldown = 0

        # Monitoring
        self.oom_detected = False
        self.oom_recovery_count = 0

    def update_vram_monitoring(self) -> Union[int, None]:
        """Update VRAM monitoring and return suggested batch size adjustment."""
        if not self.enabled:
            return None

        if torch.cuda.is_available():
            # Get current VRAM usage
            current_memory = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            current_usage_ratio = current_memory / total_memory

            self.vram_history.append(current_usage_ratio)

            # Check for OOM detection (usually handled by exception, but monitor anyway)
            if current_usage_ratio > 0.95:
                logger.warning(
                    f"Automation {self.name}: High VRAM usage detected ({current_usage_ratio:.2%})"
                )

            # Only adjust every adjustment_frequency iterations to avoid thrashing
            if self.adjustment_cooldown > 0:
                self.adjustment_cooldown -= 1
                return None

            # Calculate suggested batch size adjustment
            suggested_adjustment = self._calculate_batch_adjustment(current_usage_ratio)

            if suggested_adjustment != 0:
                self.adjustment_cooldown = self.adjustment_frequency
                return suggested_adjustment

        return None

    def _calculate_batch_adjustment(self, current_usage_ratio: float) -> int:
        """Calculate suggested batch size adjustment."""
        if self.current_batch_size is None:
            return 0

        target_usage = self.target_vram_usage

        # If significantly under target, try to increase batch size
        if current_usage_ratio < target_usage - self.safety_margin:
            if self.current_batch_size < self.max_batch_size:
                # Calculate how much we can increase
                available_memory_ratio = target_usage - current_usage_ratio
                suggested_increase = min(
                    2, int(available_memory_ratio / 0.1)
                )  # Increase by 1-2
                return suggested_increase

        # If significantly over target, decrease batch size
        elif current_usage_ratio > target_usage + self.safety_margin:
            if self.current_batch_size > self.min_batch_size:
                # Aggressive decrease if way over target
                if current_usage_ratio > target_usage + 0.1:
                    return -2  # Decrease by 2
                else:
                    return -1  # Decrease by 1

        return 0

    def set_current_batch_size(self, batch_size: int) -> None:
        """Set the current batch size for monitoring."""
        self.current_batch_size = batch_size

    def handle_oom_recovery(self, new_batch_size: int) -> None:
        """Handle OOM recovery and adjust batch size."""
        self.oom_detected = True
        self.oom_recovery_count += 1

        logger.warning(
            f"Automation {self.name}: OOM detected, adjusting batch size to {new_batch_size}"
        )

        # Reduce batch size more aggressively after OOM
        safe_batch_size = max(self.min_batch_size, new_batch_size // 2)
        self.current_batch_size = safe_batch_size

        # Set longer cooldown after OOM
        self.adjustment_cooldown = self.adjustment_frequency * 2

        # Record the adjustment
        self.record_adjustment(
            "batch_size", new_batch_size, safe_batch_size, "OOM recovery"
        )


@AUTOMATION_REGISTRY.register()
class AdaptiveGradientClipping(TrainingAutomationBase):
    """
    Adaptive Gradient Clipping

    Monitors gradient norms and automatically adjusts clipping thresholds
    to prevent exploding gradients while maintaining learning effectiveness.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Configuration
        self.initial_threshold = config.get("initial_threshold", 1.0)
        self.min_threshold = config.get("min_threshold", 0.1)
        self.max_threshold = config.get("max_threshold", 10.0)
        self.adjustment_factor = config.get("adjustment_factor", 1.2)
        self.monitoring_frequency = config.get("monitoring_frequency", 10)
        self.gradient_history_size = config.get("gradient_history_size", 100)

        # State tracking
        self.current_threshold = self.initial_threshold
        self.gradient_history = deque(maxlen=self.gradient_history_size)
        self.adjustment_cooldown = 0
        self.exploding_gradient_count = 0

        # Statistics
        self.total_gradients = 0
        self.clipped_gradients = 0

    def update_gradient_monitoring(self, gradients: list[torch.Tensor]) -> Union[float, None]:
        """Update gradient monitoring and return suggested threshold."""
        if not self.enabled:
            return None

        # Calculate current gradient norm
        if not gradients:
            return None

        total_norm = torch.sqrt(
            sum(torch.sum(g**2) for g in gradients if g is not None)
        )
        gradient_norm = float(total_norm.item())

        self.gradient_history.append(gradient_norm)
        self.total_gradients += 1

        # Check for exploding gradients
        if gradient_norm > self.current_threshold * 2:
            self.exploding_gradient_count += 1
            logger.warning(
                f"Automation {self.name}: Exploding gradient detected (norm: {gradient_norm:.4f})"
            )

        # Only adjust every monitoring_frequency iterations
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return None

        # Calculate suggested threshold adjustment
        suggested_threshold = self._calculate_threshold_adjustment(gradient_norm)

        if suggested_threshold != self.current_threshold:
            old_threshold = self.current_threshold
            self.current_threshold = suggested_threshold
            self.adjustment_cooldown = self.monitoring_frequency

            reason = (
                "exploding gradients"
                if gradient_norm > old_threshold * 2
                else "optimization"
            )
            self.record_adjustment(
                "grad_clip_threshold", old_threshold, suggested_threshold, reason
            )

            return suggested_threshold

        return None

    def _calculate_threshold_adjustment(self, current_norm: float) -> float:
        """Calculate suggested gradient clipping threshold."""
        if len(self.gradient_history) < 20:
            return self.current_threshold

        # Get recent gradient statistics
        recent_norms = list(self.gradient_history)[-20:]
        avg_norm = sum(recent_norms) / len(recent_norms)
        max_norm = max(recent_norms)

        # Adjust threshold based on gradient statistics
        if max_norm > self.current_threshold:
            # Increase threshold to accommodate larger gradients
            new_threshold = min(self.max_threshold, max_norm * 1.1)
            return new_threshold
        elif avg_norm < self.current_threshold * 0.3:
            # Decrease threshold if gradients are consistently small
            new_threshold = max(self.min_threshold, self.current_threshold * 0.8)
            return new_threshold

        return self.current_threshold

    def get_clipping_threshold(self) -> float:
        """Get current gradient clipping threshold."""
        return self.current_threshold


@AUTOMATION_REGISTRY.register()
class IntelligentEarlyStopping(TrainingAutomationBase):
    """
    Intelligent Early Stopping

    Monitors validation metrics and training progress to intelligently
    determine when to stop training to prevent overfitting and save compute.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Configuration
        self.patience = config.get("patience", 2000)
        self.min_improvement = config.get("min_improvement", 0.001)
        self.min_epochs = config.get("min_epochs", 1000)
        self.min_iterations = config.get("min_iterations", 5000)
        self.monitor_metric = config.get("monitor_metric", "val/psnr")
        self.best_metric_value = None
        self.best_iteration = 0
        self.patience_counter = 0

        # Metrics tracking
        self.metric_history = deque(maxlen=500)
        self.training_loss_history = deque(maxlen=200)

        # Additional stopping criteria
        self.max_no_improvement = config.get("max_no_improvement", self.patience)
        self.improvement_threshold = config.get("improvement_threshold", 0.002)

        # Training phase detection
        self.warmup_iterations = config.get("warmup_iterations", 1000)
        self.convergence_detected = False

    def update_training_monitoring(self, loss_value: float, iteration: int) -> None:
        """Update training loss monitoring."""
        if not self.enabled:
            return

        self.training_loss_history.append(loss_value)

        # Check for convergence based on training loss
        if (
            iteration > self.warmup_iterations
            and len(self.training_loss_history) >= 100
        ):
            recent_losses = list(self.training_loss_history)[-50:]
            loss_trend = self._calculate_loss_trend(recent_losses)

            if abs(loss_trend) < 0.001:  # Very small improvement
                logger.info(
                    f"Automation {self.name}: Training loss convergence detected"
                )
                self.convergence_detected = True

    def update_validation_monitoring(
        self, metrics: dict[str, float], iteration: int
    ) -> tuple[bool, str]:
        """
        Update validation monitoring and return (should_stop, reason).

        Returns:
            Tuple of (should_stop, reason) where reason explains why stopping was triggered
        """
        if not self.enabled:
            return False, ""

        # Check minimum iteration requirement
        if iteration < self.min_iterations:
            return False, ""

        # Extract the monitored metric
        metric_value = None
        for key in [self.monitor_metric, "psnr", "ssim", "val/psnr", "val/ssim"]:
            if key in metrics:
                metric_value = metrics[key]
                break

        if metric_value is None:
            return False, ""

        self.metric_history.append(metric_value)

        # Update best metric tracking
        if (
            self.best_metric_value is None
            or metric_value > self.best_metric_value + self.min_improvement
        ):
            self.best_metric_value = metric_value
            self.best_iteration = iteration
            self.patience_counter = 0
            logger.debug(
                f"Automation {self.name}: New best {self.monitor_metric}: {metric_value:.4f}"
            )
        else:
            self.patience_counter += 1

        # Check for early stopping conditions
        should_stop, reason = self._check_stopping_conditions(iteration, metric_value)

        if should_stop:
            logger.info(f"Automation {self.name}: Early stopping triggered - {reason}")

        return should_stop, reason

    def _check_stopping_conditions(
        self, iteration: int, current_metric: float
    ) -> tuple[bool, str]:
        """Check various early stopping conditions."""

        # Primary patience-based stopping
        if self.patience_counter >= self.patience:
            return (
                True,
                f"no improvement in {self.monitor_metric} for {self.patience} iterations",
            )

        # Convergence detection
        if self.convergence_detected and self.patience_counter >= self.patience // 2:
            return True, "training convergence detected with plateau"

        # Validation metric degradation check
        if len(self.metric_history) >= 10:
            recent_metrics = list(self.metric_history)[-10:]
            if all(m <= current_metric + self.min_improvement for m in recent_metrics):
                return True, f"{self.monitor_metric} consistently plateauing"

        # Loss-accuracy divergence check (overfitting detection)
        if len(self.training_loss_history) >= 50 and len(self.metric_history) >= 10:
            recent_loss_trend = self._calculate_loss_trend(
                list(self.training_loss_history)[-25:]
            )
            recent_metric_trend = self._calculate_metric_trend(
                list(self.metric_history)[-10:]
            )

            if recent_loss_trend < -0.001 and recent_metric_trend < -0.001:
                return (
                    True,
                    "loss improving while validation metric degrading (overfitting)",
                )

        return False, ""

    def _calculate_loss_trend(self, losses: list[float]) -> float:
        """Calculate trend in loss values (negative = improving)."""
        if len(losses) < 2:
            return 0.0

        # Simple linear trend calculation
        n = len(losses)
        x = list(range(n))
        y = losses

        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_metric_trend(self, metrics: list[float]) -> float:
        """Calculate trend in validation metrics (positive = improving)."""
        if len(metrics) < 2:
            return 0.0

        # Simple linear trend calculation
        n = len(metrics)
        x = list(range(n))
        y = metrics

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator


def create_automation(
    automation_type: str, config: dict[str, Any]
) -> TrainingAutomationBase:
    """
    Create an automation instance.

    Args:
        automation_type: Type of automation to create
        config: Configuration dictionary for the automation

    Returns:
        TrainingAutomationBase instance
    """
    return AUTOMATION_REGISTRY.get(automation_type)(config)


def get_available_automations() -> list[str]:
    """Get list of available automation types."""
    return list(AUTOMATION_REGISTRY._obj_to_name.keys())


class TrainingAutomationManager:
    """
    Manager class for coordinating multiple training automations.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.automations: dict[str, TrainingAutomationBase] = {}
        self.enabled_automations = []

        # Initialize automations
        for automation_type, automation_config in config.items():
            if isinstance(automation_config, dict) and automation_config.get(
                "enabled", False
            ):
                try:
                    automation = create_automation(automation_type, automation_config)
                    self.automations[automation_type] = automation
                    self.enabled_automations.append(automation_type)
                    logger.info(f"Initialized automation: {automation_type}")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize automation {automation_type}: {e}"
                    )

    def update_iteration(self, iteration: int) -> None:
        """Update iteration for all automations."""
        for automation in self.automations.values():
            automation.update_iteration(iteration)

    def get_automation_stats(self) -> dict[str, Any]:
        """Get statistics from all automations."""
        stats = {}
        for name, automation in self.automations.items():
            stats[name] = {
                "enabled": automation.enabled,
                "iteration": automation.iteration,
                "adjustments": automation.adjustment_count,
            }
        return stats


# Convenience function for easy integration
def setup_training_automations(opt: ReduxOptions) -> Union[TrainingAutomationManager, None]:
    """
    Set up training automations from ReduxOptions.

    Args:
        opt: ReduxOptions instance containing automation configuration

    Returns:
        TrainingAutomationManager instance or None if no automations enabled
    """
    automation_config = getattr(opt.train, "training_automations", None)

    if not automation_config or not automation_config.get("enabled", False):
        return None

    return TrainingAutomationManager(automation_config)
