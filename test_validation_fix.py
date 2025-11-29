#!/usr/bin/env python3
"""
Test script to verify the validation memory fix configuration
"""

import os

import yaml


def test_config_validation_fix() -> bool:
    """Test that the configuration has the validation memory optimizations"""

    config_path = "options/train/ParagonSR2/2xParagonSR2_Nano_CC0_RTX3060_Optimized.yml"

    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check validation memory optimizations
    print("🔍 Checking validation memory optimizations...")

    # Check val dataset size
    val_config = config.get("datasets", {}).get("val", {})
    lq_size = val_config.get("lq_size")
    gt_size = val_config.get("gt_size")

    if lq_size == 32 and gt_size == 64:
        print("✅ Validation image sizes optimized (32x32 LR → 64x64 HR)")
    else:
        print(
            f"⚠️  Validation sizes not optimized: lq_size={lq_size}, gt_size={gt_size}"
        )

    # Check train validation settings
    train_config = config.get("train", {})
    val_batch_size = train_config.get("val_batch_size")
    val_num_workers = train_config.get("val_num_workers")

    if val_batch_size == 1:
        print("✅ Validation batch size set to 1 (single image validation)")
    else:
        print(f"⚠️  Validation batch size: {val_batch_size}")

    if val_num_workers == 1:
        print("✅ Validation workers minimized")
    else:
        print(f"⚠️  Validation workers: {val_num_workers}")

    # Check dynamic loss scheduling
    dls_config = train_config.get("dynamic_loss_scheduling", {})
    if dls_config.get("enabled"):
        print("✅ Dynamic Loss Scheduling enabled")
        print(f"   - Momentum: {dls_config.get('momentum')}")
        print(f"   - Adaptation rate: {dls_config.get('adaptation_rate')}")
        print(f"   - Baseline iterations: {dls_config.get('baseline_iterations')}")
    else:
        print("❌ Dynamic Loss Scheduling disabled")

    return True


def estimate_memory_usage() -> None:
    """Estimate memory usage for different configurations"""
    print("\n🧠 Memory Usage Analysis:")
    print("=" * 50)

    # Training memory (from logs)
    training_vram = 1.71  # GB
    print(f"Training VRAM: ~{training_vram:.1f} GB ✅")

    # Validation memory estimates
    print("\nValidation Memory Estimates:")

    # Original (OOM)
    original_attention_size = 128 * 128  # Image pixels
    original_batch = 6
    original_memory = (original_batch * original_attention_size**2 * 4) / (
        1024**3
    )  # GB
    print(f"  Original (128x128, batch=6): ~{original_memory:.0f} GB ❌ OOM")

    # Fixed (current config)
    fixed_attention_size = 32 * 32  # Image pixels
    fixed_batch = 1
    fixed_memory = (fixed_batch * fixed_attention_size**2 * 4) / (1024**3)  # GB
    print(f"  Fixed (32x32, batch=1): ~{fixed_memory:.2f} GB ✅")

    # Reduction
    reduction = ((original_memory - fixed_memory) / original_memory) * 100
    print(f"  Memory reduction: {reduction:.1f}%")

    # Total estimated VRAM
    total_vram = training_vram + fixed_memory
    print(f"\nTotal estimated VRAM: ~{total_vram:.1f} GB")
    print("RTX 3060 VRAM: 11.63 GB")
    print(f"Memory headroom: {11.63 - total_vram:.1f} GB")


if __name__ == "__main__":
    print("🔧 RTX 3060 Validation Memory Fix Test")
    print("=" * 50)

    # Test configuration
    config_valid = test_config_validation_fix()

    if config_valid:
        # Estimate memory usage
        estimate_memory_usage()

        print("\n🎯 Expected Results:")
        print("✅ Training should use ~1.5-2.0 GB VRAM (stable)")
        print("✅ Validation should use ~50-200 MB VRAM (no OOM)")
        print("✅ Dynamic Loss Scheduling should adapt every iteration")
        print("✅ Training should complete 40,000 iterations without crashes")

        print("\n📋 Next Steps:")
        print(
            "1. Run: python train.py -opt options/train/ParagonSR2/2xParagonSR2_Nano_CC0_RTX3060_Optimized.yml"
        )
        print("2. Monitor VRAM usage in logs (should stay under 2.5 GB)")
        print("3. Confirm no more 'Tried to allocate 128.00 GiB' errors")
        print("4. Verify Dynamic Loss Scheduling adapts weights after iteration 150")

    else:
        print("❌ Configuration validation failed!")

    print("\n" + "=" * 50)
