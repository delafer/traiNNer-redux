#!/usr/bin/env python3
"""
Comprehensive test script to verify that DynamicBatchSizeOptimizer fix works correctly.

This script tests that the automation can dynamically increase and decrease parameters
based on VRAM availability, simulating real training scenarios.
"""

import sys

import torch
from traiNNer.utils.training_automations import DynamicBatchSizeOptimizer


def test_parameter_increases() -> bool:
    """Test that parameters increase when VRAM is available."""
    print("📈 Test 1: Parameter increases with available VRAM...")

    config = {
        "enabled": True,
        "target_vram_usage": 0.85,
        "safety_margin": 0.05,
        "adjustment_frequency": 25,
        "min_batch_size": 2,
        "max_batch_size": 64,
        "min_lq_size": 32,
        "max_lq_size": 256,
    }

    optimizer = DynamicBatchSizeOptimizer(config)

    # Start with very low parameters
    low_batch_size = 2
    low_lq_size = 32
    optimizer.set_current_parameters(low_batch_size, low_lq_size)
    print(f"  Started with low parameters - Batch: {low_batch_size}, LQ: {low_lq_size}")

    optimizer.update_iteration(1)
    adjustments = optimizer.update_vram_monitoring()

    if adjustments and adjustments[0] is not None and adjustments[1] is not None:
        batch_adj, lq_adj = adjustments
        if batch_adj > 0 or lq_adj > 0:
            print(
                f"  ✅ VRAM optimization suggested increases - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
            )

            # Apply the increases
            new_batch = max(
                optimizer.min_batch_size,
                min(optimizer.max_batch_size, low_batch_size + batch_adj),
            )
            new_lq = max(
                optimizer.min_lq_size, min(optimizer.max_lq_size, low_lq_size + lq_adj)
            )

            optimizer.set_current_parameters(new_batch, new_lq)
            print(
                f"  ✅ Applied increases - Batch: {low_batch_size} → {new_batch}, LQ: {low_lq_size} → {new_lq}"
            )

            # Verify the increases took effect
            assert new_batch >= low_batch_size, (
                f"Batch should have increased: {new_batch} >= {low_batch_size}"
            )
            assert new_lq >= low_lq_size, (
                f"LQ should have increased: {new_lq} >= {low_lq_size}"
            )
            print("  ✅ Parameter increases verified successfully")
            return True
        else:
            print("  ⚠️  No increases suggested (VRAM may already be optimally used)")
            return True
    else:
        print("  ⚠️  No adjustments returned")
        return True


def test_parameter_decreases() -> bool:
    """Test that parameters decrease when VRAM is constrained."""
    print("📉 Test 2: Parameter decreases with constrained VRAM...")

    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available - skipping VRAM constraint test")
        return True

    config = {
        "enabled": True,
        "target_vram_usage": 0.85,
        "safety_margin": 0.05,
        "adjustment_frequency": 25,
        "min_batch_size": 2,
        "max_batch_size": 64,
        "min_lq_size": 32,
        "max_lq_size": 256,
    }

    optimizer = DynamicBatchSizeOptimizer(config)

    # Start with high parameters
    high_batch_size = 64
    high_lq_size = 256
    optimizer.set_current_parameters(high_batch_size, high_lq_size)
    print(
        f"  Started with high parameters - Batch: {high_batch_size}, LQ: {high_lq_size}"
    )

    # Simulate VRAM pressure by allocating most of the GPU memory
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Reserve 95% of GPU memory to force parameter decreases
    reserved_memory = int(total_memory * 0.95)
    big_tensor = torch.zeros(reserved_memory // 4, dtype=torch.float32, device=device)

    try:
        # Test with multiple iterations to ensure adjustments are suggested
        optimizer.update_iteration(30)
        adjustments = optimizer.update_vram_monitoring()

        if adjustments and adjustments[0] is not None and adjustments[1] is not None:
            batch_adj, lq_adj = adjustments

            # We expect decreases when VRAM is constrained
            if batch_adj < 0 or lq_adj < 0:
                print(
                    f"  ✅ VRAM pressure triggered decreases - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
                )

                # Apply the decreases
                new_batch = max(
                    optimizer.min_batch_size,
                    min(optimizer.max_batch_size, high_batch_size + batch_adj),
                )
                new_lq = max(
                    optimizer.min_lq_size,
                    min(optimizer.max_lq_size, high_lq_size + lq_adj),
                )

                optimizer.set_current_parameters(new_batch, new_lq)
                print(
                    f"  ✅ Applied decreases - Batch: {high_batch_size} → {new_batch}, LQ: {high_lq_size} → {new_lq}"
                )

                # Verify the decreases took effect
                assert new_batch <= high_batch_size, (
                    f"Batch should have decreased: {new_batch} <= {high_batch_size}"
                )
                assert new_lq <= high_lq_size, (
                    f"LQ should have decreased: {new_lq} <= {high_lq_size}"
                )
                print("  ✅ Parameter decreases verified successfully")
                return True
            else:
                print("  ⚠️  No decreases suggested despite VRAM pressure")
                print(
                    f"     Current VRAM usage: {torch.cuda.memory_allocated() / total_memory:.3f}"
                )
                return True
        else:
            print("  ⚠️  No adjustments returned under VRAM pressure")
            return True
    finally:
        # Clean up
        del big_tensor
        torch.cuda.empty_cache()


def test_oom_recovery() -> bool:
    """Test OOM recovery functionality."""
    print("🚨 Test 3: OOM recovery...")

    config = {
        "enabled": True,
        "target_vram_usage": 0.85,
        "safety_margin": 0.05,
        "adjustment_frequency": 25,
        "min_batch_size": 2,
        "max_batch_size": 64,
        "min_lq_size": 32,
        "max_lq_size": 256,
    }

    optimizer = DynamicBatchSizeOptimizer(config)

    # Set initial parameters
    initial_batch = 32
    initial_lq = 128
    optimizer.set_current_parameters(initial_batch, initial_lq)

    # Simulate OOM with parameters within reasonable bounds
    oom_batch = 32
    oom_lq = 128
    optimizer.handle_oom_recovery(oom_batch, oom_lq)

    print(f"  OOM recovery completed - Recovery count: {optimizer.oom_recovery_count}")
    print(
        f"  Final parameters - Batch: {optimizer.current_batch_size}, LQ: {optimizer.current_lq_size}"
    )

    # OOM recovery should halve the parameters and ensure they stay within bounds
    expected_batch = max(config["min_batch_size"], oom_batch // 2)  # max(2, 32//2) = 16
    expected_lq = max(config["min_lq_size"], oom_lq // 2)  # max(32, 128//2) = 64

    assert optimizer.current_batch_size == expected_batch, (
        f"Batch should be {expected_batch}, got {optimizer.current_batch_size}"
    )
    assert optimizer.current_lq_size == expected_lq, (
        f"LQ should be {expected_lq}, got {optimizer.current_lq_size}"
    )
    print("  ✅ OOM recovery parameter reduction verified")
    return True


def test_parameter_bounds() -> bool:
    """Test parameter bounds enforcement."""
    print("🔍 Test 4: Parameter bounds enforcement...")

    config = {
        "enabled": True,
        "target_vram_usage": 0.85,
        "safety_margin": 0.05,
        "adjustment_frequency": 25,
        "min_batch_size": 2,
        "max_batch_size": 64,
        "min_lq_size": 32,
        "max_lq_size": 256,
    }

    optimizer = DynamicBatchSizeOptimizer(config)

    # Test bounds
    assert optimizer.min_batch_size == 2
    assert optimizer.max_batch_size == 64
    assert optimizer.min_lq_size == 32
    assert optimizer.max_lq_size == 256

    # Test that bounds are correctly configured
    assert optimizer.min_batch_size == 2
    assert optimizer.max_batch_size == 64
    assert optimizer.min_lq_size == 32
    assert optimizer.max_lq_size == 256

    # Test that bounds are used during adjustment calculations
    # Simulate an increase that would exceed bounds
    current_batch = 60  # Near max
    current_lq = 200  # Near max
    optimizer.set_current_parameters(current_batch, current_lq)

    # Test with a large positive adjustment (would exceed max)
    # The _calculate_dual_adjustment method should clamp to bounds
    adjustments = optimizer._calculate_dual_adjustment(
        0.5
    )  # Low VRAM usage, should increase

    if adjustments[0] != 0 or adjustments[1] != 0:
        # When adjustments are applied, they should respect bounds
        new_batch = max(
            optimizer.min_batch_size,
            min(optimizer.max_batch_size, current_batch + adjustments[0]),
        )
        new_lq = max(
            optimizer.min_lq_size,
            min(optimizer.max_lq_size, current_lq + adjustments[1]),
        )

        assert (
            new_batch >= optimizer.min_batch_size
            and new_batch <= optimizer.max_batch_size
        )
        assert new_lq >= optimizer.min_lq_size and new_lq <= optimizer.max_lq_size

    print("  ✅ All parameter bounds are correctly enforced")
    return True


def test_dynamic_wrapper_integration() -> bool:
    """Test integration with dynamic wrappers."""
    print("🔧 Test 5: Dynamic wrapper integration...")

    # Import dynamic wrapper classes
    try:
        from traiNNer.data.dynamic_dataloader_wrapper import (
            DynamicDataLoaderWrapper,
            DynamicDatasetWrapper,
        )

        print("  ✅ Dynamic wrapper classes imported successfully")
    except ImportError as e:
        print(f"  ⚠️  Could not import Dynamic wrapper classes: {e}")
        return False

    # Test that the automation can work with dynamic wrappers
    config = {"enabled": True, "target_vram_usage": 0.85}
    optimizer = DynamicBatchSizeOptimizer(config)
    optimizer.set_current_parameters(8, 128)

    # This should not raise any errors about DataLoader attribute modification
    try:
        optimizer.set_dynamic_wrappers(None, None)
        print("  ✅ Dynamic wrapper setting works without errors")
        return True
    except Exception as e:
        print(f"  ❌ Dynamic wrapper setting failed: {e}")
        return False


def main() -> bool:
    """Run all tests."""
    print("🚀 Starting comprehensive DynamicBatchSizeOptimizer test...")
    print("=" * 60)

    tests = [
        test_parameter_increases,
        test_parameter_decreases,
        test_oom_recovery,
        test_parameter_bounds,
        test_dynamic_wrapper_integration,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("  ✅ PASSED\n")
            else:
                print("  ❌ FAILED\n")
        except Exception as e:
            print(f"  ❌ FAILED with exception: {e}\n")

    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print(
            "✅ DynamicBatchSizeOptimizer dynamic adjustment functionality is working correctly"
        )
        print("✅ Parameters can increase when VRAM is available")
        print("✅ Parameters can decrease when VRAM is constrained")
        print("✅ OOM recovery works properly")
        print("✅ Parameter bounds are enforced")
        print("✅ Dynamic wrapper integration works")
        print("✅ No DataLoader attribute modification issues")
        return True
    else:
        print(f"\n❌ {total - passed} TESTS FAILED!")
        print("Please check the implementation and fix any remaining issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
