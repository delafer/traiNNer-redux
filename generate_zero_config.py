#!/usr/bin/env python3
"""
Zero-Config Training Generator for traiNNer-redux

This script demonstrates the simplest possible way to generate training configurations
using the zero-config system. Just specify your data paths and architecture!

Usage:
    python generate_zero_config.py

Author: Philip Hofmann
"""

import os

import yaml
from traiNNer.utils.zero_config_training import create_zero_config_training


def main() -> None:
    """Generate zero-config training configuration."""

    print("🚀 Zero-Config Training Configuration Generator")
    print("=" * 50)

    # User configuration (only 4 things needed!)
    config = {
        # 🎯 Architecture to train
        "architecture": "paragonsr2_nano",
        # 📁 Dataset paths (update these to your actual paths)
        "dataset_gt_path": "/home/phips/Documents/dataset/cc0/hr",
        "dataset_lq_path": "/home/phips/Documents/dataset/cc0/lr_x2_bicubic_aa",
        "val_gt_path": "/home/phips/Documents/dataset/cc0/val_hr",
        "val_lq_path": "/home/phips/Documents/dataset/cc0/val_lr_x2_bicubic_aa",
        # Optional: Custom overrides (leave empty for full auto)
        "custom_overrides": {
            # Example: Override specific settings if needed
            # "train": {
            #     "total_iter": 60000,  # Custom training duration
            # }
        },
    }

    print("📋 Configuration:")
    print(f"   Architecture: {config['architecture']}")
    print(f"   Train HR: {config['dataset_gt_path']}")
    print(f"   Train LR: {config['dataset_lq_path']}")
    print(f"   Val HR: {config['val_gt_path']}")
    print(f"   Val LR: {config['val_lq_path']}")
    print()

    # Generate zero-config training setup
    print("🔄 Generating zero-config training configuration...")

    try:
        training_config = create_zero_config_training(
            architecture=config["architecture"],
            dataset_gt_path=config["dataset_gt_path"],
            dataset_lq_path=config["dataset_lq_path"],
            val_gt_path=config["val_gt_path"],
            val_lq_path=config["val_lq_path"],
            custom_overrides=config["custom_overrides"] or None,
        )

        print("✅ Configuration generated successfully!")
        print()

        # Show what's been auto-configured
        print("🎯 Auto-Configured Settings:")
        print(
            f"   Batch Size: {training_config['datasets']['train']['batch_size_per_gpu']}"
        )
        print(f"   Total Iterations: {training_config['train']['total_iter']}")
        print(f"   Learning Rate: {training_config['train']['optim_g']['lr']}")
        print(f"   AMP Enabled: {training_config['use_amp']}")
        print(f"   BF16 Enabled: {training_config['amp_bf16']}")
        print(
            f"   Automations: {training_config['train']['training_automations']['enabled']}"
        )

        # Show active automations
        active_automations = []
        for name, settings in training_config["train"]["training_automations"].items():
            if isinstance(settings, dict) and settings.get("enabled", False):
                active_automations.append(name)

        print(f"   Active Automations: {', '.join(active_automations)}")
        print()

        # Save configuration
        output_file = "generated_zero_config.yml"
        print(f"💾 Saving configuration to: {output_file}")

        with open(output_file, "w") as f:
            yaml.dump(training_config, f, default_flow_style=False, sort_keys=False)

        print("✅ Configuration saved!")
        print()
        print("🚀 Ready to train! Run:")
        print(f"   python train.py --opt {output_file}")
        print()
        print("📊 Hardware Detection Summary:")

        # Show hardware detection results
        from traiNNer.utils.hardware_detection import HardwareDetector

        detector = HardwareDetector()
        tier = detector.get_hardware_tier()

        print(f"   Hardware Tier: {tier.upper()}")
        if detector.gpu_info["available"]:
            print(
                f"   GPU: {detector.gpu_info['names'][0] if detector.gpu_info['names'] else 'Unknown'}"
            )
            print(f"   VRAM: {detector.gpu_info['total_vram_gb']:.1f}GB")
        else:
            print("   GPU: Not available (CPU training)")

        print(f"   CPU Cores: {detector.cpu_info['logical_cores']}")
        print(f"   RAM: {detector.memory_info['total_gb']:.1f}GB")

    except Exception as e:
        print(f"❌ Error generating configuration: {e}")
        print("Please check your dataset paths and try again.")


def show_examples() -> None:
    """Show usage examples."""
    print("📚 Usage Examples:")
    print("=" * 50)

    print("\n1. Basic Usage (just specify paths):")
    print("""
config = create_zero_config_training(
    architecture="paragonsr2_nano",
    dataset_gt_path="/path/to/hr/images",
    dataset_lq_path="/path/to/lr/images"
)
""")

    print("2. With Validation Data:")
    print("""
config = create_zero_config_training(
    architecture="paragonsr2_nano",
    dataset_gt_path="/path/to/hr/train",
    dataset_lq_path="/path/to/lr/train",
    val_gt_path="/path/to/hr/val",
    val_lq_path="/path/to/lr/val"
)
""")

    print("3. With Custom Overrides (optional):")
    print("""
config = create_zero_config_training(
    architecture="paragonsr2_nano",
    dataset_gt_path="/path/to/hr/images",
    dataset_lq_path="/path/to/lr/images",
    custom_overrides={
        "train": {
            "total_iter": 60000,  # Override training duration
        }
    }
)
""")

    print("4. Hardware Detection Only:")
    print("""
from traiNNer.utils.hardware_detection import print_hardware_report
print_hardware_report()  # Shows your hardware and recommendations
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        show_examples()
    else:
        main()
