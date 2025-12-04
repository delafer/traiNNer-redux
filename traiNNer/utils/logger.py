import datetime
import logging
import os
import shutil
import subprocess
import time
from logging import Logger
from os import path as osp
from pathlib import Path
from typing import Any

import torch
from rich.logging import RichHandler
from rich.markup import escape
from torch.utils.tensorboard.writer import SummaryWriter

from traiNNer.utils.dist_util import get_dist_info, master_only
from traiNNer.utils.misc import free_space_gb_str
from traiNNer.utils.redux_options import ReduxOptions

initialized_logger = {}
logger_log_file = {}


class AvgTimer:
    def __init__(self, window: int = 200) -> None:
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start_time = None
        self.tic = None
        self.toc = None
        self.start()

    def start(self) -> None:
        self.start_time = self.tic = time.time()

    def record(self) -> None:
        if self.tic is None:
            raise ValueError("Must start timing before recording")
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self) -> float:
        return self.current_time

    def get_avg_time(self) -> float:
        return self.avg_time


class MessageLogger:
    """Enhanced Message logger for comprehensive training parameter monitoring.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Experiment name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iterations.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iteration. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default: None.
    """

    def __init__(
        self,
        opt: ReduxOptions,
        start_iter: int = 1,
        tb_logger: SummaryWriter | None = None,
    ) -> None:
        assert opt.logger is not None
        assert opt.train is not None

        self.exp_name = opt.name
        self.interval = opt.logger.print_freq
        self.start_iter = start_iter
        self.accum_iters = opt.datasets["train"].accum_iter
        self.max_iters = opt.train.total_iter
        self.use_tb_logger = opt.logger.use_tb_logger
        self.tb_logger = tb_logger

        # Enhanced training configuration tracking
        self.network_config = self._extract_network_config(opt)
        self.training_config = self._extract_training_config(opt)
        self.automation_config = self._extract_automation_config(opt)
        self.loss_config = self._extract_loss_config(opt)

        self.start_time = time.time()
        self.logger = get_root_logger()

        # Log initial configuration
        self._log_initial_configuration()

    def _extract_network_config(self, opt: ReduxOptions) -> dict[str, Any]:
        """Extract network configuration for logging."""
        config = {}
        if opt.network_g:
            config["type"] = opt.network_g.get("type", "unknown")
            config["scale"] = opt.scale
        return config

    def _extract_training_config(self, opt: ReduxOptions) -> dict[str, Any]:
        """Extract training configuration for logging."""
        config = {}
        if opt.train:
            config["total_iter"] = opt.train.total_iter
            config["ema_decay"] = getattr(opt.train, "ema_decay", 0)
            config["grad_clip"] = getattr(opt.train, "grad_clip", False)
        if opt.datasets and "train" in opt.datasets:
            train_dataset = opt.datasets["train"]
            config["patch_size"] = getattr(train_dataset, "gt_size", "unknown")
            config["batch_size"] = getattr(train_dataset, "batch_size", "unknown")
            config["accum_iter"] = getattr(train_dataset, "accum_iter", 1)
        return config

    def _extract_automation_config(self, opt: ReduxOptions) -> dict[str, Any]:
        """Extract automation configuration for logging."""
        config = {}
        if opt.train:
            automations = getattr(opt.train, "training_automations", None)
            if automations:
                # Check if any automation is enabled by looking for individual enabled flags
                enabled_count = 0
                for automation_type, automation_config in automations.items():
                    if isinstance(automation_config, dict):
                        is_enabled = automation_config.get("enabled", False)
                        config[automation_type] = is_enabled
                        if is_enabled:
                            enabled_count += 1
                        # Store the full config for debugging
                        config[f"{automation_type}_config"] = automation_config

                config["enabled"] = enabled_count > 0
                config["enabled_count"] = enabled_count
        return config

    def _extract_loss_config(self, opt: ReduxOptions) -> dict[str, Any]:
        """Extract loss configuration for logging."""
        config = {}
        if opt.train and opt.train.losses:
            config["types"] = [loss.get("type", "unknown") for loss in opt.train.losses]
            config["weights"] = [
                loss.get("loss_weight", 0) for loss in opt.train.losses
            ]
        return config

    def _log_initial_configuration(self) -> None:
        """Log initial training configuration."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 ENHANCED TRAINING LOGGING ENABLED")
        self.logger.info("=" * 80)

        # Network configuration
        self.logger.info(
            f"📊 Network: {self.network_config.get('type', 'unknown')} "
            f"(scale: {self.network_config.get('scale', 'unknown')}x)"
        )

        # Training configuration
        training_info = []
        if self.training_config.get("patch_size") != "unknown":
            training_info.append(f"patch: {self.training_config['patch_size']}")
        if self.training_config.get("batch_size") != "unknown":
            training_info.append(f"batch: {self.training_config['batch_size']}")
        if self.training_config.get("accum_iter", 1) > 1:
            training_info.append(f"accum: {self.training_config['accum_iter']}")
        training_info.append(f"scale: {self.network_config.get('scale', 'unknown')}x")

        self.logger.info(f"⚙️  Config: {', '.join(training_info)}")

        # Loss configuration
        if self.loss_config.get("types"):
            loss_types = [
                f"{t}({w:.2e})"
                for t, w in zip(
                    self.loss_config["types"], self.loss_config["weights"], strict=False
                )
            ]
            self.logger.info(f"🎯 Losses: {', '.join(loss_types)}")

        # Automation configuration
        if self.automation_config.get("enabled"):
            enabled_automations = []
            for automation_name, is_enabled in self.automation_config.items():
                if (
                    is_enabled
                    and not automation_name.endswith("_config")
                    and automation_name not in ["enabled", "enabled_count"]
                ):
                    # Clean up automation names for display
                    clean_name = automation_name.replace("_", " ").title()
                    enabled_automations.append(clean_name)

            if enabled_automations:
                automation_count = self.automation_config.get("enabled_count", 0)
                self.logger.info(
                    f"🤖 Automations: {', '.join(enabled_automations)} ({automation_count} enabled)"
                )
            else:
                self.logger.info("🤖 Automations: enabled (no details available)")
        else:
            self.logger.info("🤖 Automations: disabled")

        self.logger.info("=" * 80)

    def reset_start_time(self) -> None:
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars: dict[str, Any]) -> None:
        """Format enhanced logging message with comprehensive training parameters.

        Args:
            log_vars (dict): Contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iteration.
                lrs (list): List of learning rates.
                time (float): Iteration time.
                data_time (float): Data loading time for each iteration.
                training_automation_stats (dict): Training automation status and stats.
                dynamic_loss_stats (dict): Dynamic loss scheduling stats.
                gradient_stats (dict): Gradient monitoring stats.
        """
        # Extract basic training information
        epoch = log_vars.pop("epoch")
        current_iter = log_vars.pop("iter")
        lrs = log_vars.pop("lrs")

        # Extract enhanced monitoring data
        training_automation_stats = log_vars.pop("training_automation_stats", {})
        dynamic_loss_stats = log_vars.pop("dynamic_loss_stats", {})
        gradient_stats = log_vars.pop("gradient_stats", {})

        # Construct enhanced training log message
        message = f"[epoch:{epoch:4,d}, iter:{current_iter:8,d}, lr:("
        message += ", ".join([f"{v:.3e}" for v in lrs]) + ")] "

        # Training configuration display
        config_parts = []
        if self.network_config.get("type"):
            config_parts.append(f"net: {self.network_config['type']}")
        if self.training_config.get("patch_size") != "unknown":
            config_parts.append(f"patch: {self.training_config['patch_size']}")
        if self.training_config.get("batch_size") != "unknown":
            config_parts.append(f"batch: {self.training_config['batch_size']}")
        if self.network_config.get("scale"):
            config_parts.append(f"scale: {self.network_config['scale']}x")

        if config_parts:
            message += f"[{', '.join(config_parts)}] "

        # performance, eta
        if "time" in log_vars.keys():
            iter_time = 1 / (log_vars.pop("time") * self.accum_iters)
            log_vars.pop("data_time")

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

            message += f"[perf: {iter_time:.3f} it/s, eta: {eta_str}] "

        # Enhanced VRAM monitoring
        current_vram = torch.cuda.memory_allocated() / (1024**3)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        vram_cached = torch.cuda.memory_reserved() / (1024**3)
        message += f"[VRAM: {current_vram:.2f} GB, peak: {peak_vram:.2f} GB, cached: {vram_cached:.2f} GB] "

        # Enhanced performance monitoring
        if "time" in log_vars.keys():
            # Calculate additional performance metrics
            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            throughput = (
                self.training_config.get("batch_size", 1) / time_sec_avg
                if time_sec_avg > 0
                else 0
            )
            message += f"[throughput: {throughput:.2f} samples/s] "

        # Comprehensive loss monitoring
        loss_balance_info = self._format_comprehensive_loss_info(
            log_vars, dynamic_loss_stats
        )
        if loss_balance_info:
            message += f"[{loss_balance_info}] "

        # Enhanced gradient monitoring
        gradient_info = self._format_enhanced_gradient_info(gradient_stats, log_vars)
        if gradient_info:
            message += f"[{gradient_info}] "

        # Automation status with enhanced metrics
        automation_info = self._format_enhanced_automation_info(
            training_automation_stats
        )
        if automation_info:
            message += f"[{automation_info}] "

        # Training stability indicators
        stability_info = self._format_training_stability_info(log_vars, gradient_stats)
        if stability_info:
            message += f"[{stability_info}] "

        # Log losses and other training metrics
        loss_vars = {}
        other_vars = {}

        # Separate losses from other metrics
        for k, v in log_vars.items():
            if k.startswith("l_") or "loss" in k.lower():
                loss_vars[k] = v
            else:
                other_vars[k] = v

        # Log loss values
        if loss_vars:
            loss_message_parts = []
            for k, v in loss_vars.items():
                if isinstance(v, float):
                    loss_message_parts.append(f"{k}: {v:.3e}")
                else:
                    loss_message_parts.append(f"{k}: {v:.4e}")
            message += f"[{', '.join(loss_message_parts)}] "

        # Log other variables
        for k, v in other_vars.items():
            message += f"{k}: {v:.4e} "

        # Log to tensorboard with enhanced categorization
        self._log_to_tensorboard(
            log_vars,
            loss_vars,
            other_vars,
            dynamic_loss_stats,
            gradient_stats,
            training_automation_stats,
            current_iter,
        )

        # Log the final constructed message
        self.logger.info(message, extra={"markup": False})

    def _format_comprehensive_loss_info(
        self, log_vars: dict[str, Any], dynamic_loss_stats: dict[str, Any]
    ) -> str:
        """Format comprehensive loss information for enhanced logging."""
        parts = []

        # Extract and categorize all loss types
        content_losses = []
        gan_losses = []
        perceptual_losses = []
        regularization_losses = []
        total_loss = None

        for k, v in log_vars.items():
            if k.startswith("l_") or "loss" in k.lower():
                try:
                    loss_val = abs(float(v))
                    if "total" in k.lower():
                        total_loss = loss_val
                    elif (
                        "content" in k.lower() or "l1" in k.lower() or "l2" in k.lower()
                    ):
                        content_losses.append((k, loss_val))
                    elif "gan" in k.lower() or "discriminator" in k.lower():
                        gan_losses.append((k, loss_val))
                    elif "perceptual" in k.lower() or "lpips" in k.lower():
                        perceptual_losses.append((k, loss_val))
                    elif "reg" in k.lower() or "weight_decay" in k.lower():
                        regularization_losses.append((k, loss_val))
                except (ValueError, TypeError):
                    continue

        # Calculate loss ratios and balance metrics
        if content_losses and gan_losses:
            content_sum = sum(loss for _, loss in content_losses)
            gan_sum = sum(loss for _, loss in gan_losses)
            if content_sum > 0:
                balance_ratio = gan_sum / content_sum
                parts.append(f"loss_ratio: {balance_ratio:.2f}")

        # Show primary loss components
        if total_loss:
            parts.append(f"total: {total_loss:.3e}")
        if content_losses:
            primary_content = min(content_losses, key=lambda x: x[1])
            parts.append(f"content: {primary_content[1]:.3e}")
        if gan_losses:
            primary_gan = min(gan_losses, key=lambda x: x[1])
            parts.append(f"gan: {primary_gan[1]:.3e}")
        if perceptual_losses:
            primary_perc = min(perceptual_losses, key=lambda x: x[1])
            parts.append(f"perc: {primary_perc[1]:.3e}")

        # Dynamic loss scheduler information
        if dynamic_loss_stats.get("current_weights"):
            weight_parts = []
            for loss_name, weight in dynamic_loss_stats["current_weights"].items():
                if isinstance(weight, (int, float)):
                    weight_parts.append(f"{loss_name.split('_')[-1]}: {weight:.2f}")
            if weight_parts:
                parts.append(f"dyn_weights: {', '.join(weight_parts[:2])}")

        return ", ".join(parts) if parts else ""

    def _format_enhanced_gradient_info(
        self, gradient_stats: dict[str, Any], log_vars: dict[str, Any]
    ) -> str:
        """Format enhanced gradient monitoring information."""
        parts = []

        # Basic gradient norm
        if gradient_stats.get("grad_norm_g"):
            grad_norm = gradient_stats["grad_norm_g"]
            parts.append(f"grad_norm: {grad_norm:.3f}")

            # Gradient health indicator
            if grad_norm < 0.01:
                parts.append("grad_health: low")
            elif grad_norm > 10.0:
                parts.append("grad_health: high")
            else:
                parts.append("grad_health: normal")

        # Gradient clipping information
        if gradient_stats.get("grad_clip_threshold"):
            parts.append(f"clip_thresh: {gradient_stats['grad_clip_threshold']:.3f}")

        # Check for gradient explosion/vanishing patterns
        if "l_loss" in log_vars or "total_loss" in log_vars:
            # This would need to be enhanced with historical data for pattern detection
            pass

        return ", ".join(parts) if parts else ""

    def _format_enhanced_automation_info(
        self, training_automation_stats: dict[str, Any]
    ) -> str:
        """Format enhanced training automation status information."""
        parts = []

        # Show automation system status
        if self.automation_config.get("enabled"):
            parts.append("auto: ON")
        else:
            parts.append("auto: OFF")

        # Detailed automation metrics
        if training_automation_stats:
            for automation_name, stats in training_automation_stats.items():
                if isinstance(stats, dict) and stats.get("enabled"):
                    # VRAM optimizer status
                    if automation_name == "dynamic_batch_size_optimizer":
                        if "vram" in stats:
                            vram_stats = stats["vram"]
                            if vram_stats.get("current_batch_size"):
                                parts.append(
                                    f"batch: {vram_stats['current_batch_size']}"
                                )
                            if vram_stats.get("current_lq_size"):
                                parts.append(
                                    f"lq_size: {vram_stats['current_lq_size']}"
                                )
                            if vram_stats.get("target_usage"):
                                parts.append(
                                    f"vram_target: {vram_stats['target_usage']:.0%}"
                                )

                    # Gradient clipping status
                    elif automation_name == "adaptive_gradient_clipping":
                        if "current_threshold" in stats:
                            parts.append(f"grad_clip: {stats['current_threshold']:.3f}")

                    # Learning rate scheduler status
                    elif automation_name == "intelligent_learning_rate_scheduler":
                        if stats.get("lr_multipliers"):
                            avg_multiplier = sum(
                                stats["lr_multipliers"].values()
                            ) / len(stats["lr_multipliers"])
                            parts.append(f"lr_mult: {avg_multiplier:.2f}")

                    # Early stopping status
                    elif automation_name == "intelligent_early_stopping":
                        if "patience_counter" in stats:
                            parts.append(f"patience: {stats['patience_counter']}")

        return ", ".join(parts) if len(parts) > 1 else ""

    def _format_training_stability_info(
        self, log_vars: dict[str, Any], gradient_stats: dict[str, Any]
    ) -> str:
        """Format training stability indicators."""
        parts = []

        # Loss volatility indicator (would need historical data for full implementation)
        if "l_loss" in log_vars:
            # Placeholder for loss volatility calculation
            parts.append("stability: monitoring")

        # Gradient stability
        if gradient_stats.get("grad_norm_g"):
            grad_norm = gradient_stats["grad_norm_g"]
            if grad_norm < 0.001:
                parts.append("gradient: vanishing")
            elif grad_norm > 100.0:
                parts.append("gradient: exploding")
            else:
                parts.append("gradient: stable")

        # VRAM stability
        current_vram = torch.cuda.memory_allocated() / (1024**3)
        if current_vram < 1.0:
            parts.append("vram: low")
        elif current_vram > 10.0:
            parts.append("vram: high")
        else:
            parts.append("vram: stable")

        return ", ".join(parts) if parts else ""

    def _format_gradient_info(self, gradient_stats: dict[str, Any]) -> str:
        """Format gradient monitoring information."""
        parts = []

        if gradient_stats.get("grad_norm_g"):
            parts.append(f"grad_norm: {gradient_stats['grad_norm_g']:.2f}")

        if gradient_stats.get("grad_clip_threshold"):
            parts.append(f"clip_thresh: {gradient_stats['grad_clip_threshold']:.2f}")

        return ", ".join(parts) if parts else ""

    def _format_automation_info(self, training_automation_stats: dict[str, Any]) -> str:
        """Format training automation status information."""
        parts = []

        # Check for automation system status
        if training_automation_stats:
            for automation_name, stats in training_automation_stats.items():
                if isinstance(stats, dict) and stats.get("enabled"):
                    # Show key automation metrics
                    if automation_name == "adaptive_gradient_clipping":
                        if "current_threshold" in stats:
                            parts.append(f"grad_clip: {stats['current_threshold']:.2f}")
                    elif automation_name == "intelligent_learning_rate_scheduler":
                        if "lr_multipliers" in stats:
                            avg_multiplier = sum(
                                stats["lr_multipliers"].values()
                            ) / max(len(stats["lr_multipliers"]), 1)
                            parts.append(f"lr_mult: {avg_multiplier:.2f}")
                    elif automation_name == "dynamic_batch_size_optimizer":
                        if "suggested_batch_size" in stats:
                            parts.append(f"batch_opt: {stats['suggested_batch_size']}")

        # Show automation status with icons
        automation_status = []
        if self.automation_config.get("enabled"):
            automation_status.append("ON")
        else:
            automation_status.append("OFF")

        if parts:
            automation_status.extend(parts)

        return f"auto: {', '.join(automation_status)}" if automation_status else ""

    def _log_to_tensorboard(
        self,
        log_vars: dict[str, Any],
        loss_vars: dict[str, Any],
        other_vars: dict[str, Any],
        dynamic_loss_stats: dict[str, Any],
        gradient_stats: dict[str, Any],
        training_automation_stats: dict[str, Any],
        current_iter: int,
    ) -> None:
        """Log comprehensive training metrics to tensorboard with enhanced categorization."""
        if self.tb_logger is None:
            return

        # Enhanced loss logging with categorization
        for k, v in loss_vars.items():
            label = f"losses/{k}"
            value = (
                float(v)
                if isinstance(v, (int, float))
                else v.to(dtype=torch.float32).detach()
            )
            self.tb_logger.add_scalar(label, value, current_iter)

            # Also log to categorized loss groups
            if "content" in k.lower() or "l1" in k.lower() or "l2" in k.lower():
                self.tb_logger.add_scalar("loss_groups/content", value, current_iter)
            elif "gan" in k.lower():
                self.tb_logger.add_scalar("loss_groups/gan", value, current_iter)
            elif "perceptual" in k.lower() or "lpips" in k.lower():
                self.tb_logger.add_scalar("loss_groups/perceptual", value, current_iter)
            elif "total" in k.lower():
                self.tb_logger.add_scalar("loss_groups/total", value, current_iter)

        # Enhanced metrics logging
        for k, v in other_vars.items():
            label = f"metrics/{k}"
            value = (
                float(v)
                if isinstance(v, (int, float))
                else v.to(dtype=torch.float32).detach()
            )
            self.tb_logger.add_scalar(label, value, current_iter)

            # Categorize validation metrics
            if k.startswith("val/"):
                metric_name = k.replace("val/", "")
                self.tb_logger.add_scalar(
                    f"validation/{metric_name}", value, current_iter
                )

        # Enhanced VRAM monitoring
        current_vram = torch.cuda.memory_allocated() / (1024**3)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        vram_cached = torch.cuda.memory_reserved() / (1024**3)

        self.tb_logger.add_scalar("system/vram_current_gb", current_vram, current_iter)
        self.tb_logger.add_scalar("system/vram_peak_gb", peak_vram, current_iter)
        self.tb_logger.add_scalar("system/vram_cached_gb", vram_cached, current_iter)

        # VRAM utilization percentage
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_utilization = (current_vram / total_vram) * 100
            self.tb_logger.add_scalar(
                "system/vram_utilization_percent", vram_utilization, current_iter
            )

        # Enhanced performance metrics
        if "time" in log_vars:
            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            throughput = (
                self.training_config.get("batch_size", 1) / time_sec_avg
                if time_sec_avg > 0
                else 0
            )
            self.tb_logger.add_scalar(
                "performance/throughput_samples_per_sec", throughput, current_iter
            )
            self.tb_logger.add_scalar(
                "performance/avg_iter_time_sec", time_sec_avg, current_iter
            )

        # Enhanced dynamic loss scheduler logging
        if dynamic_loss_stats:
            if "current_weights" in dynamic_loss_stats:
                for loss_name, weight in dynamic_loss_stats["current_weights"].items():
                    label = f"dynamic_loss/{loss_name}_weight"
                    value = (
                        float(weight)
                        if isinstance(weight, (int, float))
                        else weight.item()
                    )
                    self.tb_logger.add_scalar(label, value, current_iter)

            # Log loss balance ratios
            content_losses = [
                v
                for k, v in log_vars.items()
                if "content" in k.lower() or "l1" in k.lower()
            ]
            gan_losses = [v for k, v in log_vars.items() if "gan" in k.lower()]
            if content_losses and gan_losses:
                content_sum = sum(abs(float(v)) for v in content_losses)
                gan_sum = sum(abs(float(v)) for v in gan_losses)
                if content_sum > 0:
                    balance_ratio = gan_sum / content_sum
                    self.tb_logger.add_scalar(
                        "loss_analysis/gan_to_content_ratio",
                        balance_ratio,
                        current_iter,
                    )

        # Enhanced gradient logging with health indicators
        for stat_name, value in gradient_stats.items():
            if isinstance(value, (int, float, torch.Tensor)):
                label = f"gradients/{stat_name}"
                tensor_value = (
                    torch.tensor(value)
                    if not isinstance(value, torch.Tensor)
                    else value
                )
                final_value = (
                    float(tensor_value.detach())
                    if hasattr(tensor_value, "detach")
                    else value
                )
                self.tb_logger.add_scalar(label, final_value, current_iter)

                # Add gradient health indicators
                if stat_name == "grad_norm_g":
                    if final_value < 0.001:
                        self.tb_logger.add_scalar(
                            "gradient_health/vanishing", 1.0, current_iter
                        )
                    elif final_value > 100.0:
                        self.tb_logger.add_scalar(
                            "gradient_health/exploding", 1.0, current_iter
                        )
                    else:
                        self.tb_logger.add_scalar(
                            "gradient_health/stable", 1.0, current_iter
                        )

        # Enhanced automation stats with detailed metrics
        for automation_name, stats in training_automation_stats.items():
            if isinstance(stats, dict):
                # Log basic automation status
                self.tb_logger.add_scalar(
                    f"automation/{automation_name}/enabled",
                    1.0 if stats.get("enabled") else 0.0,
                    current_iter,
                )

                # Log detailed automation metrics
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)) and stat_name not in ["enabled"]:
                        label = f"automation/{automation_name}/{stat_name}"
                        self.tb_logger.add_scalar(label, float(value), current_iter)

                        # Log VRAM-specific metrics with additional insights
                        if (
                            automation_name == "dynamic_batch_size_optimizer"
                            and stat_name == "vram"
                        ):
                            if isinstance(value, dict):
                                for vram_metric, vram_value in value.items():
                                    if isinstance(vram_value, (int, float)):
                                        vram_label = f"automation/{automation_name}/vram_{vram_metric}"
                                        self.tb_logger.add_scalar(
                                            vram_label, float(vram_value), current_iter
                                        )

        # Training stability indicators
        stability_score = 0.0
        if gradient_stats.get("grad_norm_g"):
            grad_norm = gradient_stats["grad_norm_g"]
            if 0.01 <= grad_norm <= 10.0:
                stability_score += 0.5

        if 1.0 <= current_vram <= 10.0:  # Assuming reasonable VRAM usage
            stability_score += 0.5

        self.tb_logger.add_scalar(
            "training/stability_score", stability_score, current_iter
        )


@master_only
def init_tb_logger(log_dir: str) -> SummaryWriter:
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt: ReduxOptions) -> None:
    """We now only use wandb to sync tensorboard log."""
    import wandb  # type: ignore

    assert opt.logger is not None
    assert opt.logger.wandb is not None
    logger = get_root_logger()

    project = opt.logger.wandb.project
    resume_id = opt.logger.wandb.resume_id
    if resume_id:
        wandb_id = resume_id
        resume = "allow"
        logger.warning("Resume wandb logger with id=%s.", wandb_id)
    else:
        wandb_id = wandb.util.generate_id()  # type: ignore
        resume = "never"

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt.name,
        config=opt,  # type: ignore
        project=project,
        sync_tensorboard=True,
    )

    logger.info("Use wandb logger with id=%s; project=%s.", wandb_id, project)


def get_root_logger(
    logger_name: str = "traiNNer",
    log_level_console: int = logging.INFO,
    log_level_file: int = logging.DEBUG,
    log_file: str | None = None,
) -> Logger:
    """Get the root logger with enhanced error handling and comprehensive logging coverage.

    Enhanced features:
    - Robust path validation and sanitization
    - Multiple fallback log directories
    - Detailed error reporting for debugging
    - Enhanced security (prevent path traversal attacks)
    - Comprehensive training metrics logging

    Args:
        logger_name (str): root logger name. Default: 'traiNNer'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level_console (int): Console logging level.
        log_level_file (int): File logging level.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)

    # Check if this is a new logger initialization or an update to an existing one
    is_new_logger = logger_name not in initialized_logger

    rank, _ = get_dist_info()

    # Always set up basic console logging for new loggers
    if is_new_logger:
        logger.info(f"🔍 Logger initialization: rank={rank}, log_file={log_file}")
        format_str = "%(asctime)s %(levelname)s: %(message)s"
        rich_handler = RichHandler(
            markup=True, rich_tracebacks=True, omit_repeated_times=False
        )
        rich_handler.setLevel(log_level_console)
        logger.addHandler(rich_handler)
        logger.propagate = False

        if rank != 0:
            logger.setLevel("ERROR")
            logger.info(
                f"📝 Logger: Non-master process (rank {rank}), console-only logging"
            )
        else:
            logger.setLevel(log_level_file)
            logger.info(
                f"📝 Logger: Master process, allowing file logging at level {log_level_file}"
            )

        if log_file is not None:
            # Try multiple approaches to create log file
            success = _setup_file_logging(logger, log_file, format_str, log_level_file)
            if not success:
                # Try fallback log directory
                fallback_success = _try_fallback_logging(
                    logger, format_str, log_level_file
                )
                if fallback_success:
                    logger.warning(
                        "📝 Logger: Primary log path failed, using fallback location"
                    )
                else:
                    logger.warning(
                        "📝 Logger: All file logging attempts failed, console-only mode"
                    )
            else:
                # Store the log file info for future reference
                logger_log_file[logger_name] = log_file
        else:
            logger.info("📝 Logger: No log_file specified, console-only logging")

        initialized_logger[logger_name] = True
    # Logger already exists, but check if we need to add file logging
    elif log_file is not None and logger_name not in logger_log_file:
        logger.info("📝 Logger: Adding file logging to existing logger")
        format_str = "%(asctime)s %(levelname)s: %(message)s"
        success = _setup_file_logging(logger, log_file, format_str, log_level_file)
        if success:
            logger_log_file[logger_name] = log_file
            logger.info("📝 Logger: File logging successfully added to existing logger")
        else:
            logger.warning("📝 Logger: Failed to add file logging to existing logger")

    return logger


def _validate_and_sanitize_path(file_path: str) -> tuple[bool, str, str]:
    """Validate and sanitize file path for safe logging.

    Args:
        file_path: Raw file path to validate

    Returns:
        Tuple of (is_valid, sanitized_path, error_message)
    """
    if not file_path:
        return False, "", "Empty file path"

    # Normalize path
    try:
        normalized_path = osp.normpath(file_path)
    except Exception as e:
        return False, "", f"Path normalization failed: {e}"

    # Check for path traversal attacks (but allow absolute paths)
    if ".." in normalized_path:
        return False, "", "Invalid path: potential security risk detected"

    # Check path length (common filesystem limits)
    if len(normalized_path) > 240:
        return False, "", f"Path too long ({len(normalized_path)} chars, max 240)"

    # Check for invalid characters (cross-platform)
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
    for char in invalid_chars:
        if char in normalized_path:
            return False, "", f"Invalid character '{char}' in path"

    return True, normalized_path, ""


def _try_fallback_logging(logger: Logger, format_str: str, log_level_file: int) -> bool:
    """Try alternative log directories when primary path fails.

    Args:
        logger: Logger instance to add handler to
        format_str: Log format string
        log_level_file: Log level for file handler

    Returns:
        True if fallback logging was successful
    """
    fallback_dirs = [
        os.path.expanduser("~/trainner_logs"),
        "/tmp/trainner_logs",
        "./logs",
        ".",
    ]

    for fallback_dir in fallback_dirs:
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_log_file = os.path.join(
                fallback_dir,
                f"trainner_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )

            file_handler = logging.FileHandler(fallback_log_file, "w")
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level_file)
            logger.addHandler(file_handler)

            logger.info(
                f"✅ Logger: Fallback file logging activated at {fallback_log_file}"
            )
            return True

        except Exception as e:
            logger.warning(f"❌ Logger: Fallback directory {fallback_dir} failed: {e}")
            continue

    return False


def _setup_file_logging(
    logger: Logger, log_file: str, format_str: str, log_level_file: int
) -> bool:
    """Set up file logging with comprehensive error handling.

    Args:
        logger: Logger instance to add handler to
        log_file: Target log file path
        format_str: Log format string
        log_level_file: Log level for file handler

    Returns:
        True if file logging was successfully set up
    """
    try:
        # Validate and sanitize path
        is_valid, sanitized_path, error_msg = _validate_and_sanitize_path(log_file)
        if not is_valid:
            logger.error(f"❌ Logger: Invalid log file path: {error_msg}")
            return False

        # Ensure log directory exists with enhanced error handling
        log_dir = osp.dirname(sanitized_path)
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.debug(f"✅ Logger: Directory ready: {log_dir}")
            except PermissionError as e:
                logger.error(
                    f"❌ Logger: Permission denied creating directory {log_dir}: {e}"
                )
                return False
            except OSError as e:
                logger.error(f"❌ Logger: OS error creating directory {log_dir}: {e}")
                return False

        # Test write access before creating handler
        try:
            test_path = sanitized_path + ".test"
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
            logger.debug(f"✅ Logger: Write access verified for {sanitized_path}")
        except Exception as e:
            logger.error(
                f"❌ Logger: Write access test failed for {sanitized_path}: {e}"
            )
            return False

        # Create file handler with enhanced error handling
        try:
            file_handler = logging.FileHandler(sanitized_path, "w", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level_file)
            logger.addHandler(file_handler)
            logger.info(f"✅ Logger: File logging activated: {sanitized_path}")
            return True

        except PermissionError as e:
            logger.error(
                f"❌ Logger: Permission denied writing to {sanitized_path}: {e}"
            )
            return False
        except OSError as e:
            logger.error(
                f"❌ Logger: OS error creating file handler for {sanitized_path}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"❌ Logger: Unexpected error creating file handler for {sanitized_path}: {e}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ Logger: Unexpected error in file logging setup: {e}")
        return False


def get_env_info() -> str:
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    device_info = torch.cuda.get_device_properties(torch.cuda.current_device())

    # from traiNNer.version import __version__
    msg = r"[italic red]:rocket:  traiNNer-redux: good luck! :rocket:[/]"
    msg += (
        "\nSystem Information: "
        f"\n\tCurrent GPU: "
        f"\n\t\tName: {device_info.name}"
        f"\n\t\tTotal VRAM: {device_info.total_memory / (1024**3):.2f} GB"
        f"\n\t\tCompute Capability: {device_info.major}.{device_info.minor}"
        f"\n\t\tMultiprocessors: {device_info.multi_processor_count}"
        f"\n\tStorage:"
        f"\n\t\tFree Space: {free_space_gb_str()}"
        "\nVersion Information: "
        f"\n\ttraiNNer-redux: {log_git_status()}"
        f"\n\tPyTorch: {torch.__version__}"
        f"\n\tTorchVision: {torchvision.__version__}"
    )
    return msg


def log_git_status() -> str | None:
    logger = get_root_logger()
    if shutil.which("git") is None:
        logger.warning(
            "[yellow]Git is not installed or not available in PATH. "
            "You may have downloaded this repo as a ZIP, which is not recommended. "
            "Please install git, then clone the repo using [bold]git clone[/bold] to ensure easier updates. Please see the %s for more info.[/yellow]",
            clickable_url(
                "https://trainner-redux.readthedocs.io/en/latest/getting_started.html#initial-setup",
                "documentation",
            ),
        )
        return

    # Check if we're inside a Git repo
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "[yellow]This directory is not a Git repository. "
            "You may have downloaded this repo as a ZIP file, which is not recommended. "
            "Please clone the repo using [bold]git clone[/bold] to ensure easier updates. Please see the %s for more info.[/yellow]",
            clickable_url(
                "https://trainner-redux.readthedocs.io/en/latest/getting_started.html#initial-setup",
                "documentation",
            ),
        )
        return

    msg = ""

    try:
        # Get commit hash and date
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )

        commit_date = (
            subprocess.check_output(
                ["git", "show", "-s", "--format=%cd", "--date=iso", commit_hash],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

        # Check if working directory is clean
        status_output = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty_flag = " (dirty)" if status_output else ""

        # Check if local is behind remote
        try:
            subprocess.check_output(
                ["git", "remote", "update"], stderr=subprocess.DEVNULL
            )
            behind_output = subprocess.check_output(
                ["git", "status", "-uno"], stderr=subprocess.DEVNULL
            ).decode("utf-8")

            behind_flag = " ([green]Up to date[/green])"
            if "Your branch is behind" in behind_output:
                behind_flag = " ([yellow]updates available - please use [bold]git pull[/bold] to update[/yellow])"
        except Exception:
            behind_flag = ""

        msg += f"\n\t\tGit commit: {commit_hash}{dirty_flag}{behind_flag}"
        msg += f"\n\t\tCommit date: {commit_date}"
        return msg

    except Exception as e:
        logger.warning("Failed to get git information: %s", e)


def clickable_file_path(file_path: str | Path, display_text: str) -> str:
    file_path = str(file_path).replace(" ", "%20")
    out = f"[link=file:///{file_path}]{escape(display_text)}[/link]"
    # print(out)
    return out


def clickable_url(url: str, display_text: str) -> str:
    url = str(url).replace(" ", "%20")
    out = f"[link={url}]{escape(display_text)}[/link]"
    return out
