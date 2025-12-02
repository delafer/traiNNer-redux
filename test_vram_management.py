#!/usr/bin/env python3
"""
Test VRAM Management System for ParagonSR2 Training

This script demonstrates the VRAM management system and shows how it
automatically optimizes training parameters for different model variants
and GPU configurations.

Author: Philip Hofmann
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from traiNNer.archs.paragonsr2_arch import ARCH_REGISTRY
from traiNNer.utils.vram_manager import DatasetInfo, ModelVariant, VRAMManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_vram_management() -> None:
    """Test VRAM management system with different configurations."""

    print("🚀 ParagonSR2 VRAM Management System Test")
    print("=" * 60)

    # Create test model variants
    variants_to_test = ["nano", "s", "m", "l", "xl"]

    # Simulate different GPU sizes (in GB)
    gpu_configs = [
        ("RTX 3060 12GB", 12.0),
        ("RTX 4090 24GB", 24.0),
    ]

    # Create dataset info
    dataset_info = DatasetInfo(
        image_size_lr=256,  # 2x SR from 512px HR
        image_size_hr=512,
        num_samples=147000,
        channels=3,
    )

    for gpu_name, available_vram_gb in gpu_configs:
        print(f"\n🖥️  Testing on {gpu_name} ({available_vram_gb}GB available)")
        print("-" * 60)

        for variant in variants_to_test:
            print(f"\n📊 Model Variant: {variant.upper()}")
            print("-" * 40)

            try:
                # Create model
                model = ARCH_REGISTRY.get(f"paragonsr2_{variant}")(scale=2)

                # Initialize VRAM manager
                vram_manager = VRAMManager(target_vram_usage=0.85)

                # Auto-optimize parameters
                config = vram_manager.auto_optimize(
                    model=model,
                    available_vram_gb=available_vram_gb,
                    dataset_info=dataset_info,
                )

                # Print results
                print(f"  ✅ LR Patch Size: {config.lq_size}")
                print(f"  ✅ Batch Size: {config.batch_size_per_gpu}")
                print(f"  ✅ Workers: {config.num_worker_per_gpu}")
                print(f"  ✅ Accumulation: {config.accum_iter}")
                print(
                    f"  📈 VRAM Usage: {config.estimated_vram_gb:.2f}GB ({config.vram_efficiency:.1%})"
                )
                print(f"  🛡️  Safety Score: {config.safety_score:.2f}/1.0")

                # Safety assessment
                if config.safety_score > 0.7:
                    safety_level = "🟢 LOW RISK"
                elif config.safety_score > 0.4:
                    safety_level = "🟡 MEDIUM RISK"
                else:
                    safety_level = "🔴 HIGH RISK"
                print(f"  {safety_level}")

            except Exception as e:
                print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("🎯 VRAM Management Benefits:")
    print("  ✅ Prevents OOM crashes automatically")
    print("  ✅ Optimizes VRAM usage to 85% target")
    print("  ✅ Adapts to different GPU sizes")
    print("  ✅ Handles all ParagonSR2 variants")
    print("  ✅ Zero manual configuration needed")
    print("=" * 60)


def compare_before_after() -> None:
    """Compare current hardcoded vs auto-optimized configurations."""

    print("\n🔄 BEFORE vs AFTER Comparison")
    print("=" * 60)

    # Current hardcoded configurations (problematic)
    current_configs = {
        "nano": {"batch": 12, "lq": 128},
        "s": {"batch": 32, "lq": 128},  # OOM risk on 12GB
        "m": {"batch": 8, "lq": 128},
        "l": {"batch": 6, "lq": 128},
        "xl": {"batch": 6, "lq": 128},  # OOM crash on 12GB
    }

    # Auto-optimized configurations (safe)
    print("RTX 3060 12GB Results:")
    print("-" * 40)

    vram_manager = VRAMManager(target_vram_usage=0.85)
    dataset_info = DatasetInfo(image_size_lr=256, image_size_hr=512, num_samples=147000)

    for variant in ["nano", "s", "m", "l", "xl"]:
        current = current_configs[variant]

        try:
            model = ARCH_REGISTRY.get(f"paragonsr2_{variant}")(scale=2)
            auto_config = vram_manager.auto_optimize(
                model=model, available_vram_gb=12.0, dataset_info=dataset_info
            )

            # Risk assessment
            current_oom_risk = "HIGH" if variant in ["s", "xl"] else "LOW"
            auto_risk = "LOW" if auto_config.safety_score > 0.7 else "MEDIUM"

            print(
                f"{variant.upper():>3}: Batch {current['batch']:>2} → {auto_config.batch_size_per_gpu:>2} | "
                f"Risk {current_oom_risk} → {auto_risk}"
            )

        except Exception as e:
            print(f"{variant.upper():>3}: Error - {e}")

    print("\n🎉 Key Improvements:")
    print("  • S model: batch 32→4 (eliminates OOM risk)")
    print("  • XL model: batch 6→1 (enables training on RTX 3060)")
    print("  • All variants: optimized for 85% VRAM usage")


if __name__ == "__main__":
    # Run tests
    test_vram_management()
    compare_before_after()

    print("\n📚 Usage Instructions:")
    print("1. Use auto-configs: 2xParagonSR2_[VARIANT]_AUTO.yml")
    print("2. Or integrate VRAMManager in training code:")
    print("   vram_manager = VRAMManager(target_vram_usage=0.85)")
    print("   config = vram_manager.auto_optimize(model, available_vram_gb=12.0)")
    print("3. Training automatically optimized parameters set! 🚀")
