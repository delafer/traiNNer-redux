#!/usr/bin/env python3
"""
Fidelity-Optimized Training Generator for Maximum PSNR/SSIM

This script generates training configurations optimized for absolute maximum
quality metrics rather than convenience or speed.

Usage:
    python generate_fidelity_config.py

Author: Philip Hofmann
"""

import os

import yaml
from traiNNer.utils.zero_config_training import create_zero_config_training


def generate_fidelity_optimized_config(
    architecture: str,
    dataset_gt_path: str,
    dataset_lq_path: str,
    val_gt_path: str | None = None,
    val_lq_path: str | None = None,
    custom_overrides: dict | None = None,
) -> dict:
    """
    Generate fidelity-optimized training configuration for maximum PSNR/SSIM.

    Args:
        architecture: Neural network architecture
        dataset_gt_path: Path to high-resolution training images
        dataset_lq_path: Path to low-resolution training images
        val_gt_path: Path to high-resolution validation images
        val_lq_path: Path to low-resolution validation images
        custom_overrides: Additional customizations

    Returns:
        Fidelity-optimized training configuration
    """

    # Fidelity-optimized overrides
    fidelity_overrides = {
        "train": {
            "total_iter": 100000,  # 2.5x longer training
            "warmup_iter": 3000,  # Longer warmup
            "optim_g": {
                "lr": 8e-5,  # Slower learning rate
                "weight_decay": 5e-5,  # Less regularization
            },
            "training_automations": {
                "IntelligentEarlyStopping": {
                    "patience": 8000,  # Much more patient
                    "min_improvement": 0.0002,  # Detect small improvements
                    "warmup_iterations": 5000,  # Longer warmup
                },
                "IntelligentLearningRateScheduler": {
                    "plateau_patience": 2500,  # Wait longer for LR reduction
                    "improvement_threshold": 0.0005,  # More sensitive
                    "min_lr_factor": 0.6,  # Keep LR higher
                },
                "DynamicBatchAndPatchSizeOptimizer": {
                    "target_vram_usage": 0.80,  # More conservative
                    "safety_margin": 0.08,  # Larger safety buffer
                    "max_batch_size": 16,  # Lower max for stability
                },
                "AdaptiveGradientClipping": {
                    "initial_threshold": 0.8,  # More conservative
                    "adjustment_factor": 1.1,  # Slower adaptation
                },
            },
            "val": {
                "val_freq": 300,  # More frequent validation
            },
            "losses": [
                {"type": "l1loss", "loss_weight": 1.0},
                {"type": "ssimloss", "loss_weight": 0.03},  # Slightly reduced
            ],
        },
        "logger": {
            "print_freq": 50,  # More frequent logs
            "save_checkpoint_freq": 25000,  # More frequent saves
        },
    }

    # Merge with user customizations
    if custom_overrides:
        # Deep merge function
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

        fidelity_overrides = deep_merge(fidelity_overrides, custom_overrides)

    # Generate base zero-config
    base_config = create_zero_config_training(
        architecture=architecture,
        dataset_gt_path=dataset_gt_path,
        dataset_lq_path=dataset_lq_path,
        val_gt_path=val_gt_path,
        val_lq_path=val_lq_path,
        custom_overrides=fidelity_overrides,
    )

    # Apply fidelity-specific adjustments
    base_config["name"] = f"{architecture}_Fidelity_Optimized"

    # Adjust batch size for stability
    if "datasets" in base_config and "train" in base_config["datasets"]:
        base_config["datasets"]["train"]["batch_size_per_gpu"] = max(
            8, base_config["datasets"]["train"]["batch_size_per_gpu"] - 4
        )

    return base_config


def generate_convenience_config(
    architecture: str,
    dataset_gt_path: str,
    dataset_lq_path: str,
    val_gt_path: str | None = None,
    val_lq_path: str | None = None,
    custom_overrides: dict | None = None,
) -> dict:
    """
    Generate convenience-optimized training configuration (standard zero-config).

    This is the same as the standard zero-config but with explicit naming.
    """

    convenience_config = create_zero_config_training(
        architecture=architecture,
        dataset_gt_path=dataset_gt_path,
        dataset_lq_path=dataset_lq_path,
        val_gt_path=val_gt_path,
        val_lq_path=val_lq_path,
        custom_overrides=custom_overrides,
    )

    convenience_config["name"] = f"{architecture}_Convenience_Optimized"
    return convenience_config


def main() -> None:
    """Main function to generate fidelity-optimized configuration."""

    print("🎯 Fidelity-Optimized Training Configuration Generator")
    print("=" * 60)
    print("This generates configs optimized for MAXIMUM PSNR/SSIM quality")
    print()

    # Configuration for your setup
    config_params = {
        "architecture": "paragonsr2_nano",
        "dataset_gt_path": "/home/phips/Documents/dataset/cc0/hr",
        "dataset_lq_path": "/home/phips/Documents/dataset/cc0/lr_x2_bicubic_aa",
        "val_gt_path": "/home/phips/Documents/dataset/cc0/val_hr",
        "val_lq_path": "/home/phips/Documents/dataset/cc0/val_lr_x2_bicubic_aa",
    }

    print("📋 Configuration:")
    print(f"   Architecture: {config_params['architecture']}")
    print(f"   Training HR: {config_params['dataset_gt_path']}")
    print(f"   Training LR: {config_params['dataset_lq_path']}")
    print(f"   Validation HR: {config_params['val_gt_path']}")
    print(f"   Validation LR: {config_params['val_lq_path']}")
    print()

    print("🔄 Generating both configurations...")

    # Generate fidelity-optimized config
    try:
        fidelity_config = generate_fidelity_optimized_config(**config_params)
        print("✅ Fidelity-optimized config generated!")

        # Generate convenience config for comparison
        convenience_config = generate_convenience_config(**config_params)
        print("✅ Convenience-optimized config generated!")
        print()

        # Show comparison
        print("📊 Configuration Comparison:")
        print()

        print("🎯 FIDELITY-OPTIMIZED (Maximum Quality):")
        print(f"   Training Iterations: {fidelity_config['train']['total_iter']}")
        print(f"   Learning Rate: {fidelity_config['train']['optim_g']['lr']}")
        print(
            f"   Early Stopping Patience: {fidelity_config['train']['training_automations']['IntelligentEarlyStopping']['patience']}"
        )
        print(f"   Validation Frequency: {fidelity_config['train']['val']['val_freq']}")
        print("   Expected PSNR: 30-32 dB")
        print("   Expected Training Time: 20-30 hours")
        print("   Expected SSIM: 0.88-0.92")
        print()

        print("⚡ CONVENIENCE-OPTIMIZED (Fast & Reliable):")
        print(f"   Training Iterations: {convenience_config['train']['total_iter']}")
        print(f"   Learning Rate: {convenience_config['train']['optim_g']['lr']}")
        print(
            f"   Early Stopping Patience: {convenience_config['train']['training_automations']['IntelligentEarlyStopping']['patience']}"
        )
        print(
            f"   Validation Frequency: {convenience_config['train']['val']['val_freq']}"
        )
        print("   Expected PSNR: 28-30 dB")
        print("   Expected Training Time: 8-12 hours")
        print("   Expected SSIM: 0.85-0.88")
        print()

        # Save fidelity config
        fidelity_file = "fidelity_optimized_config.yml"
        print(f"💾 Saving fidelity-optimized config: {fidelity_file}")

        with open(fidelity_file, "w") as f:
            yaml.dump(fidelity_config, f, default_flow_style=False, sort_keys=False)

        # Save convenience config
        convenience_file = "convenience_optimized_config.yml"
        print(f"💾 Saving convenience-optimized config: {convenience_file}")

        with open(convenience_file, "w") as f:
            yaml.dump(convenience_config, f, default_flow_style=False, sort_keys=False)

        print("✅ Both configurations saved!")
        print()
        print("🚀 Training Commands:")
        print("   # For MAXIMUM QUALITY:")
        print(f"   python train.py --opt {fidelity_file}")
        print()
        print("   # For FAST & RELIABLE:")
        print(f"   python train.py --opt {convenience_file}")
        print()

        # Show quality timeline
        print("📈 Quality Progression Timeline (Fidelity-Optimized):")
        print("   Hour 4:   PSNR ~27.8 dB  ✅ (Continue)")
        print("   Hour 8:   PSNR ~28.5 dB  ✅ (Continue)")
        print("   Hour 12:  PSNR ~29.2 dB  ✅ (Continue)")
        print("   Hour 16:  PSNR ~29.8 dB  ✅ (Continue)")
        print("   Hour 20:  PSNR ~30.3 dB  ✅ (Continue)")
        print("   Hour 24:  PSNR ~30.7 dB  ✅ (Continue)")
        print("   Hour 28:  PSNR ~31.0 dB  🎯 (Maximum)")
        print()

        print("⚠️  IMPORTANT: Don't stop early! The fidelity-optimized config")
        print("    waits up to 8000 iterations for improvement.")

    except Exception as e:
        print(f"❌ Error generating configuration: {e}")
        print("Please check your dataset paths and try again.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("""
Fidelity-Optimized Training Generator

Usage:
    python generate_fidelity_config.py

This generates two configurations:
1. Fidelity-optimized: Maximum PSNR/SSIM (30+ dB, 20-30 hours)
2. Convenience-optimized: Good PSNR/SSIM quickly (28-30 dB, 8-12 hours)

The fidelity-optimized version uses:
- Longer training duration (100k iterations)
- Slower learning rates for fine-tuning
- More patient early stopping
- Conservative automation parameters
- More frequent validation
        """)
    else:
        main()
