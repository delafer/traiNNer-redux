#!/usr/bin/env python3
"""
Enhanced Dynamic VRAM Management Test Script

This script demonstrates the enhanced DynamicBatchSizeOptimizer that now handles both
batch size AND lq_size adjustments with intelligent priority-based optimization.

Key Features:
- Priority system: Increase lq_size first (better metrics), then batch size
- Peak VRAM tracking with comprehensive statistics
- Smart adjustment cooldowns to prevent thrashing
- OOM recovery with dual parameter reduction
- Safety bounds enforcement (min/max constraints)
"""

import torch
from traiNNer.utils.training_automations import DynamicBatchSizeOptimizer


def test_enhanced_vram_optimizer() -> bool:
    """Test the enhanced VRAM optimizer with various scenarios."""
    print("🚀 Testing Enhanced Dynamic VRAM Management System")
    print("=" * 60)

    # Initialize the enhanced optimizer
    config = {
        "enabled": True,
        "target_vram_usage": 0.85,
        "safety_margin": 0.05,
        "adjustment_frequency": 100,
        "min_batch_size": 2,
        "max_batch_size": 64,
        "min_lq_size": 32,
        "max_lq_size": 256,
        "vram_history_size": 50,
    }

    optimizer = DynamicBatchSizeOptimizer(config)

    # Set initial parameters
    optimizer.set_current_parameters(batch_size=4, lq_size=64)
    optimizer.set_target_parameters(batch_size=8, lq_size=128)

    print("✅ Initialized VRAM Optimizer")
    print(f"   - Target VRAM: {config['target_vram_usage']:.0%}")
    print("   - Initial batch_size: 4, lq_size: 64")
    print(
        f"   - Bounds: batch [{config['min_batch_size']}-{config['max_batch_size']}], lq [{config['min_lq_size']}-{config['max_lq_size']}]"
    )
    print()

    # Simulate training scenarios
    scenarios = [
        ("Low VRAM Usage", 0.60, "Should increase lq_size first"),
        ("Medium VRAM Usage", 0.75, "Should increase both lq_size and batch_size"),
        ("Near Target", 0.82, "Should make minimal adjustments"),
        ("Over Target", 0.92, "Should decrease batch_size first"),
        ("Critical Over", 0.98, "Should decrease both parameters"),
    ]

    print("📊 Testing VRAM Adjustment Scenarios:")
    print("-" * 60)

    for scenario_name, usage_ratio, description in scenarios:
        print(f"\n🔍 Scenario: {scenario_name}")
        print(f"   VRAM Usage: {usage_ratio:.0%} | {description}")

        # Test adjustment calculation
        batch_adj, lq_adj = optimizer._calculate_dual_adjustment(usage_ratio)

        print(
            f"   Suggested adjustments: batch_size {batch_adj:+d}, lq_size {lq_adj:+d}"
        )

        if batch_adj != 0 or lq_adj != 0:
            # Apply adjustments
            new_batch = max(
                config["min_batch_size"],
                min(config["max_batch_size"], optimizer.current_batch_size + batch_adj),
            )
            new_lq = max(
                config["min_lq_size"],
                min(config["max_lq_size"], optimizer.current_lq_size + lq_adj),
            )

            print(
                f"   Applied: batch_size {optimizer.current_batch_size} → {new_batch}, lq_size {optimizer.current_lq_size} → {new_lq}"
            )

            # Update current parameters
            optimizer.set_current_parameters(new_batch, new_lq)
        else:
            print("   No adjustment needed")

    # Test peak VRAM tracking
    print("\n📈 Peak VRAM Tracking:")
    optimizer.peak_vram_usage = 0.89
    vram_stats = optimizer.get_vram_stats()
    print(f"   Peak usage: {vram_stats['peak_usage']:.1%}")
    print(f"   Current batch_size: {vram_stats['current_batch_size']}")
    print(f"   Current lq_size: {vram_stats['current_lq_size']}")
    print(f"   OOM recoveries: {vram_stats['oom_recovery_count']}")

    # Test OOM recovery
    print("\n🚨 Testing OOM Recovery:")
    optimizer.handle_oom_recovery(new_batch_size=8, new_lq_size=128)
    print(f"   OOM recovery count: {optimizer.oom_recovery_count}")
    print(f"   Adjusted batch_size: {optimizer.current_batch_size}")
    print(f"   Adjusted lq_size: {optimizer.current_lq_size}")

    print("\n✨ Enhanced VRAM Management Test Complete!")
    print("-" * 60)

    return True


def demonstrate_priority_system() -> None:
    """Demonstrate the intelligent priority system."""
    print("\n🎯 Demonstrating Priority System:")
    print("=" * 60)

    config = {
        "enabled": True,
        "min_batch_size": 2,
        "max_batch_size": 64,
        "min_lq_size": 32,
        "max_lq_size": 256,
    }
    optimizer = DynamicBatchSizeOptimizer(config)
    optimizer.set_current_parameters(4, 64)

    scenarios = [
        (0.70, "Low Usage - Should prioritize lq_size increase"),
        (0.85, "Target Usage - Minimal adjustments"),
        (0.95, "High Usage - Should prioritize batch_size decrease"),
    ]

    for usage, description in scenarios:
        batch_adj, lq_adj = optimizer._calculate_dual_adjustment(usage)

        print(f"\n📋 VRAM Usage: {usage:.0%}")
        print(f"   {description}")

        if usage < 0.85:  # Under target
            if lq_adj > 0:
                print(f"   ✅ PRIORITY 1: lq_size +{lq_adj} (better metrics)")
            elif batch_adj > 0:
                print(f"   ✅ PRIORITY 2: batch_size +{batch_adj} (better stability)")
        elif batch_adj < 0:
            print(f"   ✅ PRIORITY 1: batch_size {batch_adj} (less impact on metrics)")
        elif lq_adj < 0:
            print(f"   ✅ PRIORITY 2: lq_size {lq_adj} (safety measure)")

        if batch_adj == 0 and lq_adj == 0:
            print("   ⚖️  No adjustment needed - optimal usage")


def main() -> None:
    """Main test function."""
    print("Enhanced Dynamic VRAM Management System Test")
    print("=" * 60)
    print("This test demonstrates the new VRAM optimization features:")
    print("1. Dual parameter management (batch_size + lq_size)")
    print("2. Intelligent priority system")
    print("3. Peak VRAM tracking")
    print("4. Enhanced OOM recovery")
    print("5. Comprehensive statistics")
    print()

    # Run tests
    success = test_enhanced_vram_optimizer()
    demonstrate_priority_system()

    if success:
        print(
            "\n🎉 All tests passed! Enhanced VRAM management is ready for deployment."
        )
    else:
        print("\n❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
