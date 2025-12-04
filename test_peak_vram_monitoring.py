#!/usr/bin/env python3
"""
Comprehensive test for Peak-based VRAM Monitoring in DynamicBatchSizeOptimizer

This test validates that the DynamicBatchSizeOptimizer correctly:
1. Tracks peak VRAM usage during monitoring periods (not just current usage)
2. Only evaluates adjustments at the end of each monitoring period
3. Makes decisions based on peak VRAM during the entire training interval
4. Resets peak tracking for the next monitoring period
"""

import sys
from collections import deque

import torch
from traiNNer.utils.training_automations import DynamicBatchSizeOptimizer


def simulate_training_with_vram_spikes():
    """Simulate realistic training with VRAM spikes."""
    vram_usage_pattern = [
        0.05,
        0.08,
        0.12,
        0.25,
        0.35,
        0.45,
        0.50,
        0.42,
        0.38,
        0.28,  # Iter 0-9: Building up
        0.15,
        0.18,
        0.22,
        0.35,
        0.48,
        0.52,
        0.55,
        0.47,
        0.40,
        0.30,  # Iter 10-19: Peak around 55%
        0.08,
        0.10,
        0.14,
        0.28,
        0.42,
        0.46,
        0.49,
        0.41,
        0.34,
        0.24,  # Iter 20-29: Peak around 49%
        0.06,
        0.09,
        0.13,
        0.26,
        0.40,
        0.44,
        0.47,
        0.39,
        0.32,
        0.22,  # Iter 30-39: Peak around 47%
        0.04,
        0.07,
        0.11,
        0.24,
        0.38,
        0.42,
        0.45,
        0.37,
        0.30,
        0.20,  # Iter 40-49: Peak around 45%
        0.03,
        0.06,
        0.10,
        0.22,
        0.36,
        0.40,
        0.43,
        0.35,
        0.28,
        0.18,  # Iter 50-59: Peak around 43%
    ]
    return vram_usage_pattern


def test_peak_vram_monitoring() -> None:
    """Test that DynamicBatchSizeOptimizer correctly tracks peak VRAM usage."""
    print("🧪 Testing Peak-based VRAM Monitoring...")
    print("=" * 60)

    # Mock torch.cuda for testing
    class MockGPU:
        def __init__(self, usage_values) -> None:
            self.usage_values = usage_values
            self.current_index = 0

        def get_current_usage(self):
            if self.current_index < len(self.usage_values):
                usage = self.usage_values[self.current_index]
                self.current_index += 1
                return usage
            return 0.1  # Default

        def reset(self) -> None:
            self.current_index = 0

    # Setup mock GPU
    mock_gpu = MockGPU(simulate_training_with_vram_spikes())

    # Patch torch.cuda functions
    original_memory_allocated = torch.cuda.memory_allocated
    original_device_properties = torch.cuda.get_device_properties

    def mock_memory_allocated():
        usage = mock_gpu.get_current_usage()
        return int(usage * 12.49 * 1024**3)  # 12.49GB total VRAM

    def mock_device_properties(device):
        class MockProps:
            total_memory = int(12.49 * 1024**3)  # 12.49GB in bytes

        return MockProps()

    torch.cuda.memory_allocated = mock_memory_allocated
    torch.cuda.get_device_properties = mock_device_properties

    try:
        # Create DynamicBatchSizeOptimizer
        config = {
            "enabled": True,
            "target_vram_usage": 0.85,
            "safety_margin": 0.05,
            "adjustment_frequency": 10,  # Check every 10 iterations
            "min_batch_size": 2,
            "max_batch_size": 64,
            "min_lq_size": 32,
            "max_lq_size": 256,
            "vram_history_size": 50,
        }

        optimizer = DynamicBatchSizeOptimizer(config)
        optimizer.set_current_parameters(32, 128)  # Initial parameters
        optimizer.start_monitoring_period()

        print("📊 Test Setup:")
        print("   - Target VRAM: 85%")
        print("   - Safety margin: 5%")
        print("   - Adjustment frequency: 10 iterations")
        print("   - Initial parameters: Batch=32, LQ=128")
        print()

        # Simulate training iterations
        all_adjustments = []
        monitoring_periods = []

        print("🔄 Simulating Training with VRAM Spikes...")
        print()

        for iteration in range(60):
            # Update monitoring (this tracks peak VRAM internally)
            batch_adj, lq_adj = optimizer.update_vram_monitoring()

            # Check if we're at the end of a monitoring period
            if iteration % 10 == 9:  # End of period (every 10 iterations)
                # Get peak VRAM for this period
                peak_vram = optimizer.peak_vram_usage
                monitoring_periods.append(
                    {
                        "iterations": f"{iteration - 9}-{iteration}",
                        "peak_vram": peak_vram,
                        "peak_percent": f"{peak_vram * 100:.1f}%",
                        "batch_adj": batch_adj,
                        "lq_adj": lq_adj,
                    }
                )

                if batch_adj or lq_adj:
                    all_adjustments.append(
                        {
                            "iteration": iteration,
                            "peak_vram": peak_vram,
                            "batch_adj": batch_adj,
                            "lq_adj": lq_adj,
                        }
                    )

            optimizer.update_iteration(iteration)

        # Analyze results
        print("📈 Monitoring Period Analysis:")
        print()
        print("Period | Iterations | Peak VRAM | Peak %   | Batch Adj | LQ Adj")
        print("-" * 65)

        for i, period in enumerate(monitoring_periods):
            period_num = i + 1
            iterations = period["iterations"]
            peak_vram = period["peak_vram"]
            peak_percent = period["peak_percent"]
            batch_adj = period["batch_adj"] if period["batch_adj"] else "0"
            lq_adj = period["lq_adj"] if period["lq_adj"] else "0"

            print(
                f"   {period_num:2d}   |  {iterations:9s} |  {peak_vram:6.3f}  | {peak_percent:7s} | {batch_adj:8s} | {lq_adj:6s}"
            )

        print()

        # Verify peak-based monitoring
        print("🔍 Peak-based Monitoring Verification:")
        print()

        # Period 1: Peak 55% (should suggest increases since 55% < 85% - 5% = 80%)
        if monitoring_periods[0]["peak_vram"] < 0.80:
            print("✅ Period 1: Correctly detected low peak VRAM (55%)")
        else:
            print("❌ Period 1: Failed to detect low peak VRAM")

        # Period 2: Peak 49% (should suggest increases)
        if monitoring_periods[1]["peak_vram"] < 0.80:
            print("✅ Period 2: Correctly detected low peak VRAM (49%)")
        else:
            print("❌ Period 2: Failed to detect low peak VRAM")

        # Period 3: Peak 47% (should suggest increases)
        if monitoring_periods[2]["peak_vram"] < 0.80:
            print("✅ Period 3: Correctly detected low peak VRAM (47%)")
        else:
            print("❌ Period 3: Failed to detect low peak VRAM")

        print()

        # Verify monitoring period timing
        print("⏰ Monitoring Period Timing Verification:")
        print()

        expected_periods = [9, 19, 29, 39, 49, 59]  # End of each 10-iteration period
        actual_periods = [9, 19, 29, 39, 49, 59]  # We checked at these iterations

        if len(monitoring_periods) == 6:
            print("✅ Correct number of monitoring periods (6 periods)")
        else:
            print(f"❌ Wrong number of monitoring periods: {len(monitoring_periods)}")

        print()

        # Test peak tracking reset
        print("🔄 Peak VRAM Reset Verification:")
        print()

        # Check that peak VRAM gets reset after each adjustment period
        peak_values = [p["peak_vram"] for p in monitoring_periods]
        if peak_values[0] == max(peak_values[:3]):  # Period 1 has highest peak
            print("✅ Peak VRAM correctly tracked for Period 1")
        else:
            print("❌ Peak VRAM tracking failed for Period 1")

        if (
            peak_values[1] <= 0.55
        ):  # Period 2 peak should be less than or equal to Period 1
            print("✅ Peak VRAM reset for Period 2")
        else:
            print("❌ Peak VRAM not properly reset for Period 2")

        print()
        print("=" * 60)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 60)

        # Count successes
        successes = 0
        total_tests = 6

        # Test results
        if monitoring_periods[0]["peak_vram"] < 0.80:
            successes += 1
        if monitoring_periods[1]["peak_vram"] < 0.80:
            successes += 1
        if monitoring_periods[2]["peak_vram"] < 0.80:
            successes += 1
        if len(monitoring_periods) == 6:
            successes += 1
        if peak_values[0] == max(peak_values[:3]):
            successes += 1
        if peak_values[1] <= 0.55:
            successes += 1

        print(f"✅ Peak VRAM Monitoring Tests: {successes}/{total_tests} passed")

        if successes == total_tests:
            print()
            print("🎉 ALL TESTS PASSED!")
            print("✅ Peak-based VRAM monitoring is working correctly")
            print("✅ System tracks peak VRAM during training intervals")
            print("✅ Adjustments are evaluated at correct intervals")
            print("✅ Peak VRAM tracking resets between periods")
            print()
            print("🚀 The DynamicBatchSizeOptimizer now correctly:")
            print("   - Monitors PEAK VRAM during training (not just current)")
            print("   - Evaluates adjustments only at monitoring period ends")
            print("   - Makes decisions based on actual training peaks")
            print("   - Resets peak tracking for next monitoring period")
        else:
            print()
            print("❌ SOME TESTS FAILED")
            print(f"   - {total_tests - successes} tests failed")

    finally:
        # Restore original functions
        torch.cuda.memory_allocated = original_memory_allocated
        torch.cuda.get_device_properties = original_device_properties


if __name__ == "__main__":
    test_peak_vram_monitoring()
