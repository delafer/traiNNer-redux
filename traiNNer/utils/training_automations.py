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
from collections.abc import Callable
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
        validate_func: Callable | None = None,
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

        # LR adjustment logging control
        self.lr_log_frequency = config.get(
            "lr_log_frequency", 100
        )  # How often to log LR adjustments
        self.lr_log_cooldown = 0

        # State tracking
        self.loss_history = deque(maxlen=200)
        self.val_metric_history = deque(maxlen=50)
        self.lr_adjustment_history = deque(maxlen=10)
        self.baseline_loss = None
        self.best_loss = float("inf")
        self.best_loss_iteration = 0
        self.plateau_counter = 0
        self.lr_multipliers = {}  # Per parameter group

        # State tracking for LR adjustment logging
        self._last_lr_adjustment_reason = None
        self._last_lr_multiplier_value = None
        self._lr_adjustment_log_count = 0

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
                    # Don't use the enhanced _apply_lr_multiplier for validation-based adjustments
                    # as they are more urgent and should be logged immediately
                    logger.info(
                        f"Automation {self.name}: Validation plateau detected, reducing learning rate"
                    )
                    # Apply the reduction directly without enhanced logging for validation triggers
                    self.lr_adjustment_history.append(0.8)
                    self._lr_adjustment_log_count += 1
                    self._last_lr_adjustment_reason = "validation plateau"
                    self._last_lr_multiplier_value = 0.8

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

        # Enhanced logging control to prevent message spam
        if self.lr_log_cooldown > 0:
            self.lr_log_cooldown -= 1
            # Still record the adjustment in history for tracking
            self.lr_adjustment_history.append(multiplier)
            return

        # Check if this is a new adjustment type or if enough time has passed
        should_log = False

        if self._last_lr_adjustment_reason != reason:
            # New type of adjustment (plateau -> divergence or vice versa)
            should_log = True
        elif self._last_lr_multiplier_value != multiplier:
            # Different multiplier value than last time
            should_log = True
        elif self._lr_adjustment_log_count == 0:
            # First adjustment ever
            should_log = True

        if should_log:
            self.lr_adjustment_history.append(multiplier)
            self._lr_adjustment_log_count += 1
            self._last_lr_adjustment_reason = reason
            self._last_lr_multiplier_value = multiplier

            logger.info(
                f"Automation {self.name}: Suggested LR multiplier {multiplier:.2f} for reason: {reason}"
            )

            # Set cooldown to prevent frequent logging of similar adjustments
            self.lr_log_cooldown = self.lr_log_frequency
        else:
            # Still record in history but don't log
            self.lr_adjustment_history.append(multiplier)

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
class DynamicBatchAndPatchSizeOptimizer(TrainingAutomationBase):
    """
    Enhanced Dynamic VRAM Optimizer

    Monitors VRAM usage and automatically adjusts lq_size and batch size to optimize
    training efficiency while preventing out-of-memory errors. Priority system:
    1. Increase lq_size first (better final metrics)
    2. Then increase batch size (better stability)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Configuration
        self.target_vram_usage = config.get("target_vram_usage", 0.85)
        self.safety_margin = config.get("safety_margin", 0.05)
        self.adjustment_frequency = config.get(
            "adjustment_frequency", 25
        )  # More responsive (was 100)

        # Batch size bounds
        self.min_batch_size = config.get("min_batch_size", 2)
        self.max_batch_size = config.get("max_batch_size", 64)

        # lq_size bounds (for 2x training)
        self.min_lq_size = config.get("min_lq_size", 32)
        self.max_lq_size = config.get("max_lq_size", 256)

        self.vram_history_size = config.get("vram_history_size", 50)

        # State tracking
        self.vram_history = deque(maxlen=self.vram_history_size)
        self.peak_vram_usage = 0.0  # Track peak VRAM usage
        self.current_batch_size = None
        self.current_lq_size = None
        self.target_batch_size = None
        self.target_lq_size = None
        self.adjustment_cooldown = 0

        # Monitoring
        self.oom_detected = False
        self.oom_recovery_count = 0
        self.vram_manager = None  # Will be set during training

        # Dynamic wrappers for real-time updates
        self.dynamic_dataloader = None
        self.dynamic_dataset = None

    def update_vram_monitoring(self) -> tuple[int | None, int | None]:
        """Update VRAM monitoring and return suggested (batch_size_adjustment, lq_size_adjustment)."""
        if not self.enabled:
            return None, None

        if torch.cuda.is_available():
            # Use PEAK VRAM measurement to match main logger (critical fix for accurate optimization)
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory

            # Calculate usage ratios - use PEAK VRAM for optimization decisions
            current_usage_ratio = current_memory / total_memory
            peak_usage_ratio = peak_memory / total_memory

            # Update peak VRAM tracking for current monitoring period
            self.peak_vram_usage = max(self.peak_vram_usage, peak_usage_ratio)
            self.vram_history.append(peak_usage_ratio)

            # Always log VRAM status for debugging (but not too frequently)
            if self.iteration % 50 == 0:
                logger.info(
                    f"Automation {self.name}: VRAM usage {current_usage_ratio:.4f} ({current_memory / 1e9:.2f}GB), "
                    f"peak: {peak_usage_ratio:.4f} ({peak_memory / 1e9:.2f}GB/{total_memory / 1e9:.2f}GB), "
                    f"target: {self.target_vram_usage:.2f}"
                )

            # Check for OOM detection (usually handled by exception, but monitor anyway)
            if current_usage_ratio > 0.95:
                logger.warning(
                    f"Automation {self.name}: High VRAM usage detected ({current_usage_ratio:.2%})"
                )

            # Only evaluate adjustments at the end of each monitoring period
            if self.adjustment_cooldown > 0:
                self.adjustment_cooldown -= 1
                return None, None

            # Check if parameters are initialized
            if self.current_batch_size is None or self.current_lq_size is None:
                logger.warning(
                    f"Automation {self.name}: Parameters not initialized - batch: {self.current_batch_size}, lq: {self.current_lq_size}"
                )
                return None, None

            # CRITICAL: Don't evaluate until actual training iterations have occurred
            # Prevent premature adjustments based on initialization VRAM (0%)
            if self.iteration < self.adjustment_frequency:
                return None, None  # Skip evaluation during initial training phase

            # Calculate suggested adjustments using PEAK VRAM from current monitoring period
            batch_adjustment, lq_adjustment = self._calculate_dual_adjustment(
                self.peak_vram_usage  # Use PEAK VRAM, not current
            )

            if batch_adjustment != 0 or lq_adjustment != 0:
                # Reset peak VRAM tracking for next monitoring period
                # Start fresh tracking from current peak
                self.peak_vram_usage = peak_usage_ratio
                self.adjustment_cooldown = self.adjustment_frequency
                logger.info(
                    f"Suggested adjustments - Batch: {batch_adjustment:+d}, LQ: {lq_adjustment:+d}"
                )

                # Reset CUDA peak memory stats immediately so we can see the effect of the adjustment
                # This is CRITICAL: otherwise max_memory_allocated() keeps returning the old high value
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                return batch_adjustment, lq_adjustment

        return None, None

    def _calculate_dual_adjustment(self, peak_usage_ratio: float) -> tuple[int, int]:
        """Calculate suggested adjustments using PEAK VRAM from monitoring period with lq_size priority, then batch_size."""
        if self.current_batch_size is None or self.current_lq_size is None:
            return 0, 0

        target_usage = self.target_vram_usage
        batch_adjustment = 0
        lq_adjustment = 0

        # Enhanced logging for debugging with peak VRAM focus
        if self.iteration % 50 == 0:
            logger.info(
                f"VRAM DEBUG: Peak usage: {peak_usage_ratio:.3f}, "
                f"Target: {target_usage:.3f}, "
                f"Safety margin: {self.safety_margin:.3f}, "
                f"Current batch: {self.current_batch_size}, "
                f"Current lq_size: {self.current_lq_size}"
            )

        # If significantly under target, try to increase parameters (MORE AGGRESSIVE)
        if peak_usage_ratio < target_usage - self.safety_margin:
            available_memory_ratio = target_usage - peak_usage_ratio

            # PRIORITY 1: Increase lq_size first (better final metrics) - MORE AGGRESSIVE
            if self.current_lq_size < self.max_lq_size:
                # Much more aggressive: increase lq_size with even small memory available
                if (
                    available_memory_ratio > 0.02
                ):  # Only 2% memory available needed (was 5%)
                    # Calculate how much we can increase lq_size - more aggressive increases
                    suggested_lq_increase = min(
                        8, max(2, int(available_memory_ratio / 0.05))
                    )  # Increase by 2-8 patch sizes (was 1-4)
                    # Ensure we don't exceed max limits
                    suggested_lq_increase = min(
                        suggested_lq_increase, self.max_lq_size - self.current_lq_size
                    )

                    if suggested_lq_increase > 0:
                        lq_adjustment = suggested_lq_increase
                        logger.info(
                            f"🔄 VRAM OPTIMIZATION (PEAK-BASED): Available memory {available_memory_ratio:.3f} ({available_memory_ratio * 100:.1f}%), "
                            f"suggesting lq_size increase of +{suggested_lq_increase} "
                            f"({self.current_lq_size} → {self.current_lq_size + suggested_lq_increase})"
                        )

            # PRIORITY 2: Then increase batch size if lq_size is already at maximum
            elif self.current_batch_size < self.max_batch_size:
                remaining_memory = target_usage - peak_usage_ratio
                if remaining_memory > 0.01:  # Even 1% memory available (was 2%)
                    suggested_batch_increase = min(
                        8, max(2, int(remaining_memory / 0.03))
                    )  # Increase by 2-8 batch sizes (was 1-4)
                    # Ensure we don't exceed max limits
                    suggested_batch_increase = min(
                        suggested_batch_increase,
                        self.max_batch_size - self.current_batch_size,
                    )

                    if suggested_batch_increase > 0:
                        batch_adjustment = suggested_batch_increase
                        logger.info(
                            f"🔄 VRAM OPTIMIZATION (PEAK-BASED): Remaining memory {remaining_memory:.3f} ({remaining_memory * 100:.1f}%), "
                            f"suggesting batch_size increase of +{suggested_batch_increase} "
                            f"({self.current_batch_size} → {self.current_batch_size + suggested_batch_increase})"
                        )

        # If significantly over target, decrease parameters (reverse priority)
        elif peak_usage_ratio > target_usage + self.safety_margin:
            # PRIORITY 1: First decrease batch size (less impact on final metrics)
            if self.current_batch_size > self.min_batch_size:
                if peak_usage_ratio > target_usage + 0.1:
                    batch_adjustment = -4  # More aggressive decrease (was -2)
                else:
                    batch_adjustment = (
                        -2
                    )  # More conservative but still significant (was -1)

                logger.info(
                    f"🔄 VRAM OPTIMIZATION (PEAK-BASED): High peak usage {peak_usage_ratio:.3f}, "
                    f"suggesting batch_size decrease of {batch_adjustment} "
                    f"({self.current_batch_size} → {self.current_batch_size + batch_adjustment})"
                )

            # PRIORITY 2: Then decrease lq_size if batch is already at minimum
            elif self.current_lq_size > self.min_lq_size:
                if peak_usage_ratio > target_usage + 0.15:
                    lq_adjustment = -4  # More aggressive lq decrease
                else:
                    lq_adjustment = -2  # Conservative lq decrease (was -1)

                logger.info(
                    f"🔄 VRAM OPTIMIZATION (PEAK-BASED): High peak usage {peak_usage_ratio:.3f}, "
                    f"suggesting lq_size decrease of {lq_adjustment} "
                    f"({self.current_lq_size} → {self.current_lq_size + lq_adjustment})"
                )

        # Log final decision (only log adjustments, not "no adjustments" to reduce spam)
        if batch_adjustment != 0 or lq_adjustment != 0:
            logger.info(
                f"🎯 VRAM OPTIMIZATION DECISION (PEAK-BASED): "
                f"Peak VRAM: {peak_usage_ratio:.3f} ({peak_usage_ratio * 100:.1f}%), "
                f"Batch adjustment: {batch_adjustment:+d}, "
                f"LQ adjustment: {lq_adjustment:+d}"
            )

        return batch_adjustment, lq_adjustment

    def set_current_parameters(self, batch_size: int, lq_size: int) -> None:
        """Set the current batch size and lq_size for monitoring."""
        self.current_batch_size = batch_size
        self.current_lq_size = lq_size

        # Log parameter initialization for debugging
        if self.enabled:
            logger.info(
                f"Automation {self.name}: Parameters initialized - Batch: {batch_size}, LQ: {lq_size}"
            )

    def set_target_parameters(self, batch_size: int, lq_size: int) -> None:
        """Set the target batch size and lq_size for optimization."""
        self.target_batch_size = batch_size
        self.target_lq_size = lq_size

    def handle_oom_recovery(self, new_batch_size: int, new_lq_size: int) -> None:
        """Handle OOM recovery and adjust both batch size and lq_size."""
        self.oom_detected = True
        self.oom_recovery_count += 1

        logger.warning(
            f"Automation {self.name}: OOM detected, adjusting batch size to {new_batch_size} and lq_size to {new_lq_size}"
        )

        # Reduce parameters more aggressively after OOM
        safe_batch_size = max(self.min_batch_size, new_batch_size // 2)
        safe_lq_size = max(self.min_lq_size, new_lq_size // 2)

        self.current_batch_size = safe_batch_size
        self.current_lq_size = safe_lq_size

        # Set longer cooldown after OOM
        self.adjustment_cooldown = self.adjustment_frequency * 2

        # Apply adjustments through dynamic wrappers if available
        if hasattr(self, "dynamic_dataloader") and self.dynamic_dataloader:
            self.dynamic_dataloader.set_batch_size(safe_batch_size)
        if hasattr(self, "dynamic_dataset") and self.dynamic_dataset:
            # Check for the correct method name (Mixin uses set_dynamic_gt_size, Wrapper uses set_gt_size)
            if hasattr(self.dynamic_dataset, "set_dynamic_gt_size"):
                self.dynamic_dataset.set_dynamic_gt_size(safe_lq_size * 2)
            else:
                self.dynamic_dataset.set_gt_size(safe_lq_size * 2)  # Assuming 2x scale

        # Record the adjustments
        self.record_adjustment(
            "batch_size", new_batch_size, safe_batch_size, "OOM recovery"
        )
        self.record_adjustment("lq_size", new_lq_size, safe_lq_size, "OOM recovery")

    def set_dynamic_wrappers(
        self, dynamic_dataloader=None, dynamic_dataset=None
    ) -> None:
        """Set dynamic wrappers for real-time parameter updates."""
        self.dynamic_dataloader = dynamic_dataloader
        self.dynamic_dataset = dynamic_dataset

        logger.info(
            f"Automation {self.name}: Dynamic wrappers set - "
            f"Dataloader: {dynamic_dataloader is not None}, "
            f"Dataset: {dynamic_dataset is not None}"
        )

    def start_monitoring_period(self) -> None:
        """Initialize peak VRAM tracking for a new monitoring period."""
        if torch.cuda.is_available():
            # Reset peak memory stats for accurate tracking in new period
            torch.cuda.reset_peak_memory_stats()

            # Initialize with PEAK VRAM to match main logger measurement
            initial_memory = torch.cuda.max_memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            initial_usage_ratio = initial_memory / total_memory
            self.peak_vram_usage = initial_usage_ratio

            logger.info(
                f"Automation {self.name}: Starting VRAM monitoring period "
                f"(adjustment_frequency: {self.adjustment_frequency} iterations). "
                f"Initial Peak VRAM: {initial_usage_ratio:.3f} ({initial_usage_ratio * 100:.1f}%)"
            )

    def get_peak_vram_usage(self) -> float:
        """Get the peak VRAM usage during training."""
        return self.peak_vram_usage

    def get_vram_stats(self) -> dict[str, Any]:
        """Get comprehensive VRAM statistics."""
        if not self.vram_history:
            return {"peak_usage": 0.0, "avg_usage": 0.0, "current_usage": 0.0}

        current_usage = self.vram_history[-1] if self.vram_history else 0.0
        avg_usage = sum(self.vram_history) / len(self.vram_history)

        return {
            "peak_usage": self.peak_vram_usage,
            "avg_usage": avg_usage,
            "current_usage": current_usage,
            "target_usage": self.target_vram_usage,
            "safety_margin": self.safety_margin,
            "current_batch_size": self.current_batch_size,
            "current_lq_size": self.current_lq_size,
            "oom_recovery_count": self.oom_recovery_count,
        }


@AUTOMATION_REGISTRY.register()
class AdaptiveGradientClipping(TrainingAutomationBase):
    """
    Adaptive Gradient Clipping - Autonomous Version

    Monitors gradient norms and automatically adjusts clipping thresholds
    to prevent exploding gradients while maintaining learning effectiveness.

    AUTONOMOUS FEATURES:
    - Auto-detects model architecture (Nano, S, etc.)
    - Auto-calibrates all parameters based on architecture
    - Learning-based optimization during training
    - Minimal user configuration required
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # TRULY AUTONOMOUS: Minimal configuration required
        # User just needs enabled: true, everything else is automatic

        # State tracking - all auto-calibrated
        self.current_threshold = None
        self.gradient_history = deque(maxlen=100)
        self.gradient_stats_history = deque(maxlen=100)
        self.adjustment_cooldown = 0
        self.exploding_gradient_count = 0

        # Autonomous calibration state
        self.auto_calibrated = False
        self.calibration_iterations = 0
        self.detected_architecture = None
        self.monitoring_frequency = 10  # Check gradients every 10 iterations
        self.autonomous_bounds = {"min_threshold": 0.1, "max_threshold": 10.0}

        # Performance tracking
        self.total_gradients = 0
        self.clipped_gradients = 0
        self.adjustment_events = 0

        # Initialize autonomous mode
        if self.enabled:
            logger.info(
                "🤖 AdaptiveGradientClipping: Autonomous mode enabled - "
                "auto-calibrating parameters based on detected architecture"
            )

    def update_gradient_monitoring(self, gradients: list[torch.Tensor]) -> float | None:
        """Autonomous gradient monitoring with auto-calibration."""
        if not self.enabled:
            return None

        if not gradients:
            return None

        # Autonomous calibration phase
        if not self.auto_calibrated:
            self._autonomous_calibration(gradients)
            return None

        # Calculate gradient statistics
        total_norm = torch.sqrt(
            sum(torch.sum(g**2) for g in gradients if g is not None)
        )
        gradient_norm = float(total_norm.item())

        # Store statistics
        gradient_stats = {
            "total_norm": gradient_norm,
            "num_parameters": len([g for g in gradients if g is not None]),
        }

        self.gradient_history.append(gradient_norm)
        self.gradient_stats_history.append(gradient_stats)
        self.total_gradients += 1

        # Enhanced logging every 100 iterations
        if self.total_gradients % 100 == 0:
            self._log_autonomous_performance()

        # Check for exploding gradients
        if gradient_norm > self.current_threshold * 2:
            self.exploding_gradient_count += 1
            logger.warning(
                f"🤖 AdaptiveGradientClipping: Exploding gradient detected "
                f"(norm: {gradient_norm:.4f}, threshold: {self.current_threshold:.4f})"
            )

        # Track clipping
        if gradient_norm > self.current_threshold:
            self.clipped_gradients += 1

        # Autonomous adjustment
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return None

        # Calculate autonomous threshold adjustment
        suggested_threshold = self._autonomous_threshold_adjustment(gradient_norm)

        if suggested_threshold != self.current_threshold:
            old_threshold = self.current_threshold
            self.current_threshold = suggested_threshold
            # Set cooldown based on detected architecture
            if self.detected_architecture == "simple":
                self.adjustment_cooldown = 50
            else:  # complex
                self.adjustment_cooldown = 75
            self.adjustment_events += 1

            logger.info(
                f"🤖 AdaptiveGradientClipping: Autonomous adjustment "
                f"from {old_threshold:.4f} to {suggested_threshold:.4f} "
                f"(grad norm: {gradient_norm:.4f})"
            )

            return suggested_threshold

        return None

    def _autonomous_calibration(self, gradients: list[torch.Tensor]) -> None:
        """Auto-detect architecture and calibrate parameters."""
        self.calibration_iterations += 1

        # Calculate initial gradient statistics for calibration
        total_norm = torch.sqrt(
            sum(torch.sum(g**2) for g in gradients if g is not None)
        )
        gradient_norm = float(total_norm.item())
        self.gradient_history.append(gradient_norm)

        # Auto-detect architecture complexity based on gradient behavior
        if len(self.gradient_history) >= 20:
            recent_norms = list(self.gradient_history)[-20:]
            gradient_variance = torch.var(torch.tensor(recent_norms)).item()
            gradient_mean = sum(recent_norms) / len(recent_norms)

            # Auto-calibrate based on detected complexity
            if (
                gradient_variance > 0.001 or gradient_mean > 0.1
            ):  # High complexity (S model)
                self._calibrate_for_complex_model()
            else:  # Low complexity (Nano model)
                self._calibrate_for_simple_model()

            self.auto_calibrated = True

            logger.info(
                f"🤖 AdaptiveGradientClipping: Auto-calibration complete. "
                f"Detected {'complex' if gradient_variance > 0.001 else 'simple'} architecture. "
                f"Threshold: {self.current_threshold:.4f}, "
                f"Monitoring freq: {self.monitoring_frequency}"
            )

    def _calibrate_for_simple_model(self) -> None:
        """Auto-calibrate for simple models (Nano)."""
        self.current_threshold = 1.0
        self.detected_architecture = "simple"
        # Store autonomous bounds
        self.autonomous_bounds = {"min_threshold": 0.1, "max_threshold": 5.0}

    def _calibrate_for_complex_model(self) -> None:
        """Auto-calibrate for complex models (S)."""
        self.current_threshold = 0.8  # More conservative for complex models
        self.detected_architecture = "complex"
        # Store autonomous bounds
        self.autonomous_bounds = {"min_threshold": 0.05, "max_threshold": 8.0}

    def _autonomous_threshold_adjustment(self, gradient_norm: float) -> float:
        """Autonomous threshold adjustment based on learning."""
        if len(self.gradient_history) < 20:
            return self.current_threshold

        # Get recent statistics for autonomous decision
        recent_norms = list(self.gradient_history)[-20:]
        avg_norm = sum(recent_norms) / len(recent_norms)
        max_norm = max(recent_norms)

        # Get autonomous bounds
        min_thresh = self.autonomous_bounds["min_threshold"]
        max_thresh = self.autonomous_bounds["max_threshold"]

        # Autonomous adjustment logic
        clipping_rate = self._calculate_clipping_rate()

        # If clipping rate is too high, increase threshold
        if clipping_rate > 0.1:  # More than 10% clipping
            return min(max_thresh, self.current_threshold * 1.2)

        # If gradients are consistently small, decrease threshold
        elif avg_norm < self.current_threshold * 0.2:  # Average is 20% of threshold
            return max(min_thresh, self.current_threshold * 0.8)

        # If max gradient approaches threshold, increase slightly
        elif max_norm > self.current_threshold * 0.8:
            return min(max_thresh, max_norm * 1.1)

        return self.current_threshold

    def _log_autonomous_performance(self) -> None:
        """Log autonomous performance metrics."""
        if len(self.gradient_history) >= 10:
            recent_norms = list(self.gradient_history)[-10:]
            avg_norm = sum(recent_norms) / len(recent_norms)
            clipping_rate = self._calculate_clipping_rate()

            logger.info(
                f"🤖 AdaptiveGradientClipping: Autonomous performance "
                f"(iter {self.total_gradients}) - "
                f"Avg gradient: {avg_norm:.6f}, "
                f"Threshold: {self.current_threshold:.4f}, "
                f"Clipping rate: {clipping_rate:.3%}, "
                f"Auto-adjusted: {self.adjustment_events}x"
            )

    def _calculate_clipping_rate(self) -> float:
        """Calculate current clipping rate."""
        if self.total_gradients == 0:
            return 0.0
        return self.clipped_gradients / self.total_gradients

    def get_clipping_threshold(self) -> float:
        """Get current gradient clipping threshold."""
        # Return safe default during autonomous calibration phase
        if self.current_threshold is None:
            return 1.0  # Safe default threshold
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

        # Enhanced convergence logging control
        self.convergence_log_cooldown = 0
        self.convergence_log_frequency = config.get(
            "convergence_log_frequency", 100
        )  # Log convergence messages every N iterations when starting convergence
        self.convergence_threshold = config.get(
            "convergence_threshold", 0.0005
        )  # More conservative threshold

        # State tracking for convergence transitions
        self._was_converged_last_check = False
        self._convergence_log_count = 0
        self._sustained_convergence_iterations = 0
        self._min_sustained_for_log = config.get(
            "min_sustained_convergence_for_log", 10
        )  # Must be sustained for at least 10 iterations before logging

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

            # Check if we are currently in a convergence state
            is_currently_converged = abs(loss_trend) < self.convergence_threshold

            # Track sustained convergence
            if is_currently_converged:
                self._sustained_convergence_iterations += 1
            else:
                self._sustained_convergence_iterations = 0

            # Enhanced convergence detection with proper state tracking
            if is_currently_converged:
                # We are in convergence state
                if not self._was_converged_last_check:
                    # We just ENTERED convergence - potential logging opportunity
                    if (
                        self._sustained_convergence_iterations
                        >= self._min_sustained_for_log
                    ):
                        if self.convergence_log_cooldown == 0:
                            logger.info(
                                f"Automation {self.name}: Training loss convergence detected "
                                f"(sustained for {self._sustained_convergence_iterations} iterations, "
                                f"loss trend: {loss_trend:.6f})"
                            )
                            self.convergence_detected = True
                            self._convergence_log_count += 1
                            # Set cooldown to prevent frequent logging during sustained convergence
                            self.convergence_log_cooldown = (
                                self.convergence_log_frequency
                            )
                        else:
                            self.convergence_log_cooldown -= 1
                # We are continuing in convergence state - just decrement cooldown
                elif self.convergence_log_cooldown > 0:
                    self.convergence_log_cooldown -= 1
            else:
                # We are NOT in convergence state - reset state tracking
                self._was_converged_last_check = False
                self._sustained_convergence_iterations = 0
                self.convergence_detected = False

            # Update convergence state for next iteration
            self._was_converged_last_check = is_currently_converged

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

        # FIXED: Improved validation metric plateau detection
        # Check for real plateaus by looking at improvements relative to the best achieved
        if len(self.metric_history) >= 15:  # Increased window for more robust detection
            recent_metrics = list(self.metric_history)[-15:]

            # Check if we've had meaningful recent improvements
            best_recent = max(
                recent_metrics[:5]
            )  # Best metric in the last 5 checkpoints
            best_ever = max(self.metric_history)  # Best metric overall
            recent_improvement = best_recent - (
                max(recent_metrics[:5])
                if len(recent_metrics) > 5
                else recent_metrics[0]
            )

            # Also check the trend of recent metrics
            if len(recent_metrics) >= 10:
                recent_trend = self._calculate_metric_trend(recent_metrics)

                # Real plateau detection: if the recent trend is flat AND
                # we haven't improved significantly from our recent best
                flat_trend_threshold = (
                    self.min_improvement * 0.5
                )  # Even more conservative
                meaningful_improvement_threshold = (
                    self.min_improvement * 2
                )  # Allow larger improvements

                if (
                    abs(recent_trend) < flat_trend_threshold
                    and best_ever - best_recent < meaningful_improvement_threshold
                ):
                    return (
                        True,
                        f"{self.monitor_metric} genuinely plateauing (flat trend, no recent gains)",
                    )

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

        # Fail-safe: Register snake_case aliases here to ensure they exist
        # regardless of how the manager was instantiated
        # Fail-safe: Register snake_case aliases here to ensure they exist
        # regardless of how the manager was instantiated
        aliases = {
            "adaptive_gradient_clipping": AdaptiveGradientClipping,
            "intelligent_learning_rate_scheduler": IntelligentLearningRateScheduler,
            "dynamic_batch_and_patch_size_optimizer": DynamicBatchAndPatchSizeOptimizer,
            "intelligent_early_stopping": IntelligentEarlyStopping,
        }

        for alias, cls in aliases.items():
            if alias not in AUTOMATION_REGISTRY:
                try:
                    AUTOMATION_REGISTRY.register(cls, name=alias)
                except Exception as e:
                    logger.warning(
                        f"Could not register alias '{alias}' in manager init: {e}"
                    )

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
            automation_stats = {
                "enabled": automation.enabled,
                "iteration": automation.iteration,
                "adjustments": automation.adjustment_count,
            }

            # Add VRAM-specific stats for DynamicBatchAndPatchSizeOptimizer
            if name == "DynamicBatchAndPatchSizeOptimizer" and hasattr(
                automation, "get_vram_stats"
            ):
                automation_stats["vram"] = automation.get_vram_stats()

            stats[name] = automation_stats
        return stats


# Convenience function for easy integration
def setup_training_automations(opt: ReduxOptions) -> TrainingAutomationManager | None:
    """
    Set up training automations from ReduxOptions.

    Args:
        opt: ReduxOptions instance containing automation configuration

    Returns:
        TrainingAutomationManager instance or None if no automations enabled
    """
    # Register snake_case aliases for compatibility with config files
    # This allows using "adaptive_gradient_clipping" in config instead of "AdaptiveGradientClipping"
    # Register snake_case aliases for compatibility with config files
    # This allows using "adaptive_gradient_clipping" in config instead of "AdaptiveGradientClipping"

    aliases = {
        "adaptive_gradient_clipping": AdaptiveGradientClipping,
        "intelligent_learning_rate_scheduler": IntelligentLearningRateScheduler,
        "dynamic_batch_and_patch_size_optimizer": DynamicBatchAndPatchSizeOptimizer,
        "intelligent_early_stopping": IntelligentEarlyStopping,
    }

    for alias, cls in aliases.items():
        if alias not in AUTOMATION_REGISTRY:
            AUTOMATION_REGISTRY.register(cls, name=alias)

    # DEBUG: Print registry keys
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"DEBUG: Registry keys after registration: {list(AUTOMATION_REGISTRY._obj_map.keys())}"
    )

    automation_config = getattr(opt.train, "training_automations", None)

    if not automation_config or not automation_config.get("enabled", False):
        return None

    return TrainingAutomationManager(automation_config)
