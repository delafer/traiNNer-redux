#!/usr/bin/env python3
"""
Zero-Config Training Module for traiNNer-redux

Provides truly automatic training configuration generation based on hardware detection
and intelligent presets. This eliminates the need for users to manually configure
training automations, making training setup completely automated.

Key Features:
- Automatic hardware detection and optimization
- Zero-config automation setup
- Architecture-aware parameter selection
- Intelligent batch size calculation
- Hardware-tier specific optimization
- Graceful degradation for limited hardware

Author: Philip Hofmann
"""

import logging
from typing import Any, Dict, Optional

from .hardware_detection import HardwareDetector, detect_optimal_automation_config

logger = logging.getLogger(__name__)


class ZeroConfigTrainingManager:
    """
    Manages zero-config training setups with automatic optimization.

    This class handles the complete automation of training configuration,
    requiring only basic information like architecture and dataset paths.
    """

    def __init__(self) -> None:
        self.hardware_detector = HardwareDetector()
        self.architecture_presets = self._load_architecture_presets()

    def _load_architecture_presets(self) -> dict[str, dict[str, Any]]:
        """Load architecture-specific configuration presets."""
        return {
            "paragonsr2_nano": {
                "base_lr": 2e-4,
                "warmup_iter": 1000,
                "total_iter": 40000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "paragonsr2_micro": {
                "base_lr": 1.5e-4,
                "warmup_iter": 1500,
                "total_iter": 50000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "paragonsr2_tiny": {
                "base_lr": 1e-4,
                "warmup_iter": 2000,
                "total_iter": 60000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "paragonsr2_xs": {
                "base_lr": 8e-5,
                "warmup_iter": 2500,
                "total_iter": 70000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "paragonsr2_s": {
                "base_lr": 5e-5,
                "warmup_iter": 3000,
                "total_iter": 80000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "paragonsr2_m": {
                "base_lr": 3e-5,
                "warmup_iter": 4000,
                "total_iter": 100000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "paragonsr2_l": {
                "base_lr": 2e-5,
                "warmup_iter": 5000,
                "total_iter": 120000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "paragonsr2_xl": {
                "base_lr": 1e-5,
                "warmup_iter": 6000,
                "total_iter": 150000,
                "loss_weights": {"l1loss": 1.0, "ssimloss": 0.05},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
            "esrgan": {
                "base_lr": 1e-4,
                "warmup_iter": 1000,
                "total_iter": 50000,
                "loss_weights": {"l1loss": 1.0, "perceptual_loss": 0.1},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "complex",
            },
            "rcan": {
                "base_lr": 3e-4,
                "warmup_iter": 1000,
                "total_iter": 60000,
                "loss_weights": {"l1loss": 1.0},
                "recommended_optimizer": "AdamW",
                "dataset_complexity": "medium",
            },
        }

    def generate_zero_config_training(
        self,
        architecture: str,
        dataset_info: dict[str, str],
        custom_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate complete zero-config training configuration.

        This is the main entry point for automatic training setup.
        Only requires architecture and dataset paths - everything else is automatic.

        Args:
            architecture: Neural network architecture name
            dataset_info: Dictionary with keys like 'train_gt', 'train_lq', 'val_gt', 'val_lq'
            custom_overrides: Optional user customizations to override automatic settings

        Returns:
            Complete training configuration ready for use
        """
        # Get architecture preset
        preset = self.architecture_presets.get(
            architecture.lower(), self.architecture_presets["paragonsr2_nano"]
        )

        # Get hardware-optimized configurations
        automations_config, optimal_batch_size, hardware_recommendations = (
            detect_optimal_automation_config(
                architecture=architecture,
                total_iterations=preset["total_iter"],
                dataset_complexity=preset["dataset_complexity"],
            )
        )

        # Generate complete configuration
        config = {
            "name": f"{architecture}_ZeroConfig",
            "scale": self._extract_scale_from_architecture(architecture),
            # System optimizations (auto-detected)
            "use_amp": hardware_recommendations.get("amp_recommendation", True),
            "amp_bf16": "bf16"
            in str(hardware_recommendations.get("mixed_precision", "fp16")),
            "use_channels_last": hardware_recommendations.get("channels_last", True),
            "fast_matmul": hardware_recommendations.get("fast_matmul", True),
            "num_gpu": "auto",
            "manual_seed": 1024,
            # Dataset configuration (user-provided)
            "datasets": {
                "train": {
                    "name": "ZeroConfig_Train",
                    "type": "pairedimagedataset",
                    "dataroot_gt": dataset_info.get("train_gt", ""),
                    "dataroot_lq": dataset_info.get("train_lq", ""),
                    "lq_size": self._auto_detect_lq_size(architecture),
                    "use_hflip": True,
                    "use_rot": True,
                    "num_worker_per_gpu": hardware_recommendations.get(
                        "num_workers", 4
                    ),
                    "batch_size_per_gpu": optimal_batch_size,
                    "accum_iter": hardware_recommendations.get("accum_iter", 1),
                    "pin_memory": hardware_recommendations.get("pin_memory", True),
                },
                "val": {
                    "name": "ZeroConfig_Val",
                    "type": "pairedimagedataset",
                    "dataroot_gt": dataset_info.get("val_gt", ""),
                    "dataroot_lq": dataset_info.get("val_lq", ""),
                    "lq_size": self._auto_detect_lq_size(architecture),
                },
            },
            # Network configuration
            "network_g": {"type": architecture},
            # Training configuration with automations
            "train": {
                "ema_decay": 0.999,
                "ema_power": 0.75,
                "grad_clip": True,
                "optim_g": {
                    "type": preset["recommended_optimizer"],
                    "lr": preset["base_lr"],
                    "weight_decay": 1e-4,
                    "betas": [0.9, 0.99],
                },
                "total_iter": preset["total_iter"],
                "warmup_iter": preset["warmup_iter"],
                # All automations - completely automatic
                "training_automations": automations_config,
                # Dynamic loss scheduling - automatic
                "dynamic_loss_scheduling": {"enabled": True, "auto_calibrate": True},
                # Loss configuration - architecture preset
                "losses": [
                    {
                        "type": next(iter(preset["loss_weights"].keys())),
                        "loss_weight": next(iter(preset["loss_weights"].values())),
                    }
                ]
                + [
                    {"type": loss_type, "loss_weight": loss_weight}
                    for loss_type, loss_weight in list(preset["loss_weights"].items())[
                        1:
                    ]
                ],
            },
            # Validation configuration - automatic
            "val": {
                "val_enabled": True,
                "val_freq": max(
                    500, preset["total_iter"] // 80
                ),  # Adaptive validation frequency
                "save_img": False,
                "metrics_enabled": True,
                "metrics": {
                    "psnr": {"type": "calculate_psnr", "crop_border": 4},
                    "ssim": {"type": "calculate_ssim", "crop_border": 4},
                },
            },
            # Logging configuration - automatic
            "logger": {
                "print_freq": 100,
                "save_checkpoint_freq": preset["total_iter"] // 2,
                "save_checkpoint_format": "safetensors",
                "use_tb_logger": True,
            },
            # Path configuration - automatic
            "path": {
                "pretrain_network_g": None,
                "strict_load_g": True,
                "resume_state": None,
            },
        }

        # Apply hardware-specific optimizations
        config.update(
            self._apply_hardware_optimizations(config, hardware_recommendations)
        )

        # Apply user customizations if provided
        if custom_overrides:
            config = self._apply_custom_overrides(config, custom_overrides)

        # Log the generated configuration
        self._log_generated_config(config, hardware_recommendations)

        return config

    def _extract_scale_from_architecture(self, architecture: str) -> int:
        """Extract scale factor from architecture name."""
        # Look for scale patterns in architecture name
        if "2x" in architecture.lower():
            return 2
        elif "3x" in architecture.lower():
            return 3
        elif "4x" in architecture.lower():
            return 4
        else:
            # Default to 2x for most SR architectures
            return 2

    def _auto_detect_lq_size(self, architecture: str) -> int:
        """Automatically detect optimal low-quality patch size."""
        # Architecture-specific LQ sizes
        arch_sizes = {
            "nano": 64,
            "micro": 96,
            "tiny": 128,
            "xs": 128,
            "s": 128,
            "m": 128,
            "l": 192,
            "xl": 192,
        }

        # Find matching architecture variant
        for variant, size in arch_sizes.items():
            if variant in architecture.lower():
                return size

        # Default fallback
        return 128

    def _apply_hardware_optimizations(
        self, config: dict[str, Any], recommendations: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply hardware-specific optimizations to configuration."""
        optimizations = {}

        # Model compilation for supported hardware
        if recommendations.get("compile_model", False):
            # This would require model compilation support in the training loop
            logger.info("Model compilation recommended for this hardware")

        # Gradient accumulation for limited VRAM
        if recommendations.get("grad_accumulation", False):
            optimizations["gradient_checkpointing"] = (
                False  # Save memory for accumulation
            )
            logger.info("Gradient accumulation enabled for limited VRAM")

        # Mixed precision recommendations
        if recommendations.get("mixed_precision"):
            logger.info(
                f"Mixed precision recommended: {recommendations['mixed_precision']}"
            )

        return optimizations

    def _apply_custom_overrides(
        self, config: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply user customizations to automatically generated configuration."""

        # Deep merge custom overrides
        def deep_merge(base: dict, override: dict) -> dict:
            for key, value in override.items():
                if (
                    isinstance(value, dict)
                    and key in base
                    and isinstance(base[key], dict)
                ):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        merged_config = deep_merge(config.copy(), overrides)

        # Log what was overridden
        for key in overrides.keys():
            logger.info(f"User override applied: {key}")

        return merged_config

    def _log_generated_config(
        self, config: dict[str, Any], hardware_recommendations: dict[str, Any]
    ) -> None:
        """Log information about the generated configuration."""
        tier = self.hardware_detector.get_hardware_tier()

        logger.info("🚀 Zero-Config Training Setup Generated!")
        logger.info(f"🏗️  Architecture: {config['network_g']['type']}")
        logger.info(f"💾 Hardware Tier: {tier.upper()}")
        logger.info(
            f"⚡ Batch Size: {config['datasets']['train']['batch_size_per_gpu']}"
        )
        logger.info(f"🎯 Total Iterations: {config['train']['total_iter']}")
        logger.info(
            f"🔧 Automations Enabled: {config['train']['training_automations']['enabled']}"
        )
        logger.info(f"💿 AMP: {config['use_amp']} (BF16: {config['amp_bf16']})")

        # Log which automations are active
        active_automations = []
        for automation_name, automation_config in config["train"][
            "training_automations"
        ].items():
            if isinstance(automation_config, dict) and automation_config.get(
                "enabled", False
            ):
                active_automations.append(automation_name)

        logger.info(f"🤖 Active Automations: {', '.join(active_automations)}")

        if hardware_recommendations.get("compile_model"):
            logger.info("⚡ Model compilation available on this hardware")


# Convenience functions for easy integration
def create_zero_config_training(
    architecture: str,
    dataset_gt_path: str,
    dataset_lq_path: str,
    val_gt_path: str | None = None,
    val_lq_path: str | None = None,
    custom_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a complete zero-config training setup with minimal user input.

    This is the simplest possible interface for training setup.

    Args:
        architecture: Architecture name (e.g., 'paragonsr2_nano')
        dataset_gt_path: Path to high-resolution training images
        dataset_lq_path: Path to low-resolution training images
        val_gt_path: Path to high-resolution validation images (optional)
        val_lq_path: Path to low-resolution validation images (optional)
        custom_overrides: Optional configuration overrides

    Returns:
        Complete training configuration dictionary
    """
    manager = ZeroConfigTrainingManager()

    dataset_info = {"train_gt": dataset_gt_path, "train_lq": dataset_lq_path}

    if val_gt_path and val_lq_path:
        dataset_info["val_gt"] = val_gt_path
        dataset_info["val_lq"] = val_lq_path

    return manager.generate_zero_config_training(
        architecture=architecture,
        dataset_info=dataset_info,
        custom_overrides=custom_overrides,
    )


def print_zero_config_example() -> None:
    """Print example usage of zero-config training."""
    print("""
=== Zero-Config Training Examples ===

Example 1: Simplest possible setup
```python
from traiNNer.utils.zero_config_training import create_zero_config_training

config = create_zero_config_training(
    architecture="paragonsr2_nano",
    dataset_gt_path="/path/to/hr/images",
    dataset_lq_path="/path/to/lr/images"
)

# Save and use the config
import yaml
with open('zero_config.yml', 'w') as f:
    yaml.dump(config, f)
```

Example 2: With validation data and custom overrides
```python
config = create_zero_config_training(
    architecture="paragonsr2_micro",
    dataset_gt_path="/path/to/hr/train",
    dataset_lq_path="/path/to/lr/train",
    val_gt_path="/path/to/hr/val",
    val_lq_path="/path/to/lr/val",
    custom_overrides={
        "train": {
            "total_iter": 60000,  # Custom training duration
        }
    }
)
```

Example 3: Hardware detection report
```python
from traiNNer.utils.hardware_detection import print_hardware_report

print_hardware_report()  # Shows your hardware and recommended settings
```

Benefits:
✅ Zero manual configuration required
✅ Hardware-optimized parameters
✅ Automatic architecture detection
✅ Intelligent automation setup
✅ VRAM-aware batch sizing
✅ Safety bounds and fallbacks
""")
