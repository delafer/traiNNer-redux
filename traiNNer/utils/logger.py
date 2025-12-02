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
        message += f"[VRAM: {current_vram:.2f} GB, peak: {peak_vram:.2f} GB] "

        # Dynamic loss balance monitoring
        loss_balance_info = self._format_loss_balance_info(log_vars, dynamic_loss_stats)
        if loss_balance_info:
            message += f"[{loss_balance_info}] "

        # Gradient monitoring
        gradient_info = self._format_gradient_info(gradient_stats)
        if gradient_info:
            message += f"[{gradient_info}] "

        # Automation status
        automation_info = self._format_automation_info(training_automation_stats)
        if automation_info:
            message += f"[{automation_info}] "

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

    def _format_loss_balance_info(
        self, log_vars: dict[str, Any], dynamic_loss_stats: dict[str, Any]
    ) -> str:
        """Format loss balance information for logging."""
        parts = []

        # Check for loss balance ratios
        content_loss = None
        gan_loss = None
        perceptual_loss = None

        for k, v in log_vars.items():
            if "content" in k.lower() or "l1" in k.lower() or "l2" in k.lower():
                content_loss = abs(float(v))
            elif "gan" in k.lower():
                gan_loss = abs(float(v))
            elif "perceptual" in k.lower() or "lpips" in k.lower():
                perceptual_loss = abs(float(v))

        if content_loss and gan_loss:
            balance_ratio = gan_loss / (content_loss + 1e-8)
            parts.append(f"loss_ratio: {balance_ratio:.2f}")

        # Add dynamic loss scheduler information
        if dynamic_loss_stats.get("current_weights"):
            weight_parts = []
            for loss_name, weight in dynamic_loss_stats["current_weights"].items():
                if isinstance(weight, (int, float)):
                    weight_parts.append(f"{loss_name.split('_')[-1]}: {weight:.2f}")
            if weight_parts:
                parts.append(
                    f"dyn_weights: {', '.join(weight_parts[:2])}"
                )  # Show max 2

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
        """Log enhanced metrics to tensorboard with proper categorization."""
        if self.tb_logger is None:
            return

        # Log loss variables
        for k, v in loss_vars.items():
            label = f"losses/{k}"
            value = (
                float(v)
                if isinstance(v, (int, float))
                else v.to(dtype=torch.float32).detach()
            )
            self.tb_logger.add_scalar(label, value, current_iter)

        # Log other variables
        for k, v in other_vars.items():
            label = f"metrics/{k}"
            value = (
                float(v)
                if isinstance(v, (int, float))
                else v.to(dtype=torch.float32).detach()
            )
            self.tb_logger.add_scalar(label, value, current_iter)

        # Log dynamic loss scheduler stats
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

        # Log gradient stats
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

        # Log automation stats
        for automation_name, stats in training_automation_stats.items():
            if isinstance(stats, dict):
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        label = f"automation/{automation_name}_{stat_name}"
                        self.tb_logger.add_scalar(label, value, current_iter)


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
    """Get the root logger.
    duf_downsample
        The logger will be initialized if it has not been initialized. By default a
        StreamHandler will be added. If `log_file` is specified, a FileHandler will
        also be added.

        Args:
            logger_name (str): root logger name. Default: 'traiNNer'.
            log_file (str | None): The log filename. If specified, a FileHandler
                will be added to the root logger.
            log_level (int): The root logger level. Note that only the process of
                rank 0 is affected, while other processes will set the level to
                "Error" and be silent most of the time.

        Returns:
            logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger
    format_str = "%(asctime)s %(levelname)s: %(message)s"
    rich_handler = RichHandler(
        markup=True, rich_tracebacks=True, omit_repeated_times=False
    )
    rich_handler.setLevel(log_level_console)
    logger.addHandler(rich_handler)
    logger.propagate = False

    rank, _ = get_dist_info()
    print(f"🔍 Logger Debug: Current rank = {rank}")
    print(f"🔍 Logger Debug: Log file = {log_file}")

    if rank != 0:
        logger.setLevel("ERROR")
        print(
            f"🔍 Logger Debug: Rank {rank != 0}, setting level to ERROR (no file logging)"
        )
    else:
        logger.setLevel(log_level_file)
        print(
            f"🔍 Logger Debug: Rank {rank}, allowing file logging at level {log_level_file}"
        )
        if log_file is not None:
            try:
                # Ensure log directory exists
                log_dir = osp.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    print(f"🔍 Logger Debug: Created/verified log directory: {log_dir}")

                # add file handler
                file_handler = logging.FileHandler(log_file, "w")
                file_handler.setFormatter(logging.Formatter(format_str))
                file_handler.setLevel(log_level_file)
                logger.addHandler(file_handler)
                print(
                    f"✅ Logger Debug: File handler added successfully for {log_file}"
                )
            except Exception as e:
                print(f"❌ Logger Debug: Failed to create file handler: {e}")
                print("📝 Logger Debug: Falling back to console-only logging")
        else:
            print("🔍 Logger Debug: No log_file specified, console-only logging")
    initialized_logger[logger_name] = True
    return logger


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
