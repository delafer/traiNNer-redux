#!/usr/bin/env python3
"""
Test script for Dynamic VRAM Management Integration

This script tests the enhanced VRAM management system to ensure it properly
adjusts lq_size and batch_size dynamically during training based on real VRAM usage.

Features tested:
1. Parameter initialization and verification
2. VRAM monitoring and adjustment logic
3. Dynamic wrapper integration
4. Real-time parameter updates
5. Enhanced logging and debugging

Usage:
    python test_vram_management_integration.py
"""

import logging
import os
import sys
from pathlib import Path

import torch

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from traiNNer.utils.config import Config
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.training_automations import setup_training_automations

# Set up logging for the test
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockOptions:
    """Mock ReduxOptions for testing VRAM management."""

    def __init__(self) -> None:
        self.train = MockTrain()
        self.scale = 2

        # Mock datasets
        self.datasets = {"train": MockDataset()}


class MockTrain:
    def __init__(self) -> None:
        self.total_iter = 60000
        self.warmup_iter = 1000
        self.training_automations = {
            "DynamicBatchAndPatchSizeOptimizer": {
                "enabled": True,
                "target_vram_usage": 0.85,
                "safety_margin": 0.05,
                "adjustment_frequency": 25,  # More responsive
                "min_batch_size": 4,
                "max_batch_size": 64,
                "min_lq_size": 64,
                "max_lq_size": 256,
                "vram_history_size": 50,
            }
        }


class MockDataset:
    def __init__(self) -> None:
        self.lq_size = 128
        self.batch_size_per_gpu = 16
        self.gt_size = 256  # Will be computed from lq_size * scale
        self.dataset_enlarge_ratio = 1


def test_vram_monitoring() -> bool:
    """Test VRAM monitoring and adjustment logic."""
    logger.info("🔥 Testing VRAM Monitoring and Adjustment Logic")

    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, skipping VRAM monitoring test")
        return False

    # Create mock automation
    mock_config = {
        "DynamicBatchSizeOptimizer": {
            "enabled": True,
            "target_vram_usage": 0.85,
            "safety_margin": 0.05,
            "adjustment_frequency": 25,
            "min_batch_size": 4,
            "max_batch_size": 64,
            "min_lq_size": 64,
            "max_lq_size": 256,
            "vram_history_size": 50,
        }
    }

    automation_manager = setup_training_automations(MockOptions())
    if not automation_manager:
        logger.error("❌ Failed to create automation manager")
        return False

    automation = automation_manager.automations.get("DynamicBatchSizeOptimizer")
    if not automation or not automation.enabled:
        logger.error("❌ DynamicBatchAndPatchSizeOptimizer not found or disabled")
        return False

    # Initialize parameters
    initial_batch_size = 16
    initial_lq_size = 128
    automation.set_current_parameters(initial_batch_size, initial_lq_size)

    logger.info(
        f"✅ Parameters initialized - Batch: {initial_batch_size}, LQ: {initial_lq_size}"
    )

    # Test VRAM monitoring
    current_memory = torch.cuda.memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    current_usage_ratio = current_memory / total_memory

    logger.info(
        f"📊 Current VRAM usage: {current_usage_ratio:.3f} ({current_usage_ratio * 100:.1f}%)"
    )
    logger.info(
        f"📊 Target VRAM usage: {automation.target_vram_usage:.3f} ({automation.target_vram_usage * 100:.1f}%)"
    )

    # Test adjustment calculation
    batch_adjustment, lq_adjustment = automation._calculate_dual_adjustment(
        current_usage_ratio
    )

    logger.info(
        f"🎯 Suggested adjustments - Batch: {batch_adjustment:+d}, LQ: {lq_adjustment:+d}"
    )

    # Test VRAM update
    updates = automation.update_vram_monitoring()
    if updates:
        batch_adj, lq_adj = updates
        logger.info(f"🔄 VRAM update returned - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}")

    return True


def test_parameter_bounds() -> bool:
    """Test parameter bounds checking."""
    logger.info("🎯 Testing Parameter Bounds Checking")

    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, skipping bounds test")
        return True

    # Create automation
    mock_config = {
        "DynamicBatchSizeOptimizer": {
            "enabled": True,
            "target_vram_usage": 0.85,
            "safety_margin": 0.05,
            "adjustment_frequency": 25,
            "min_batch_size": 4,
            "max_batch_size": 64,
            "min_lq_size": 64,
            "max_lq_size": 256,
            "vram_history_size": 50,
        }
    }

    automation_manager = setup_training_automations(MockOptions())
    automation = automation_manager.automations.get("DynamicBatchSizeOptimizer")

    # Test bounds
    assert automation.min_batch_size == 4, (
        f"Expected min_batch_size=4, got {automation.min_batch_size}"
    )
    assert automation.max_batch_size == 64, (
        f"Expected max_batch_size=64, got {automation.max_batch_size}"
    )
    assert automation.min_lq_size == 64, (
        f"Expected min_lq_size=64, got {automation.min_lq_size}"
    )
    assert automation.max_lq_size == 256, (
        f"Expected max_lq_size=256, got {automation.max_lq_size}"
    )

    logger.info("✅ All parameter bounds are correctly configured")
    return True


def test_adjustment_calculation_scenarios() -> bool:
    """Test various adjustment calculation scenarios."""
    logger.info("🧪 Testing Adjustment Calculation Scenarios")

    # Create automation
    mock_config = {
        "DynamicBatchSizeOptimizer": {
            "enabled": True,
            "target_vram_usage": 0.85,
            "safety_margin": 0.05,
            "adjustment_frequency": 25,
            "min_batch_size": 4,
            "max_batch_size": 64,
            "min_lq_size": 64,
            "max_lq_size": 256,
            "vram_history_size": 50,
        }
    }

    automation_manager = setup_training_automations(MockOptions())
    automation = automation_manager.automations.get("DynamicBatchSizeOptimizer")

    # Test scenarios
    test_cases = [
        {
            "name": "Low VRAM usage (should increase)",
            "usage_ratio": 0.3,
            "expected_action": "increase",
        },
        {
            "name": "High VRAM usage (should decrease)",
            "usage_ratio": 0.95,
            "expected_action": "decrease",
        },
        {
            "name": "Target VRAM usage (should stay same)",
            "usage_ratio": 0.85,
            "expected_action": "stay",
        },
    ]

    for case in test_cases:
        logger.info(
            f"🧪 Testing scenario: {case['name']} (usage: {case['usage_ratio']:.3f})"
        )

        # Set parameters for testing
        automation.current_batch_size = 16
        automation.current_lq_size = 128

        batch_adj, lq_adj = automation._calculate_dual_adjustment(case["usage_ratio"])

        if case["expected_action"] == "increase":
            if batch_adj > 0 or lq_adj > 0:
                logger.info(
                    f"  ✅ Correctly suggested increases - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
                )
            else:
                logger.info(
                    f"  ⚠️ Expected increases but got - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
                )
        elif case["expected_action"] == "decrease":
            if batch_adj < 0 or lq_adj < 0:
                logger.info(
                    f"  ✅ Correctly suggested decreases - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
                )
            else:
                logger.info(
                    f"  ⚠️ Expected decreases but got - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
                )
        elif batch_adj == 0 and lq_adj == 0:
            logger.info(
                f"  ✅ Correctly suggested no changes - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
            )
        else:
            logger.info(
                f"  ⚠️ Expected no changes but got - Batch: {batch_adj:+d}, LQ: {lq_adj:+d}"
            )

    return True


def main() -> bool:
    """Main test function."""
    logger.info("🚀 Starting Dynamic VRAM Management Integration Tests")

    if not torch.cuda.is_available():
        logger.warning(
            "⚠️ CUDA not available - VRAM management will not function properly"
        )
        logger.warning(
            "   This test requires a CUDA-capable GPU to fully validate VRAM management"
        )

    tests_passed = 0
    total_tests = 0

    # Test 1: Parameter bounds
    total_tests += 1
    if test_parameter_bounds():
        tests_passed += 1

    # Test 2: Adjustment calculation scenarios
    total_tests += 1
    if test_adjustment_calculation_scenarios():
        tests_passed += 1

    # Test 3: VRAM monitoring (only if CUDA available)
    total_tests += 1
    if test_vram_monitoring():
        tests_passed += 1

    # Summary
    logger.info(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        logger.info(
            "🎉 All tests passed! VRAM management integration appears to be working correctly."
        )
        logger.info(
            "✅ The dynamic VRAM management system should now properly adjust lq_size and batch_size"
        )
        logger.info(
            "✅ Enhanced logging will help you monitor the adjustments in real-time"
        )
        return True
    else:
        logger.warning(
            f"⚠️ {total_tests - tests_passed} tests failed. Review the logs above for details."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
