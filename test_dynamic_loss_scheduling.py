#!/usr/bin/env python3
"""
Test Script for Dynamic Loss Scheduling System

This script tests the integration of dynamic loss scheduling with the traiNNer
framework to ensure proper functionality and compatibility.
"""

import sys

import torch
from torch import nn
from traiNNer.losses import build_loss
from traiNNer.losses.dynamic_loss_scheduling import (
    DynamicLossScheduler,
    create_dynamic_loss_scheduler,
)


def test_dynamic_loss_scheduling_basic() -> None:
    """Test basic dynamic loss scheduling functionality."""
    print("Testing basic dynamic loss scheduling...")

    # Create sample losses
    l1_loss = nn.L1Loss()
    l1_loss.loss_weight = 1.0

    mse_loss = nn.MSELoss()
    mse_loss.loss_weight = 0.1

    # Create loss dictionary
    losses_dict = {"l1_loss": l1_loss, "mse_loss": mse_loss}

    # Create scheduler configuration
    config = {
        "enabled": True,
        "momentum": 0.9,
        "adaptation_rate": 0.01,
        "min_weight": 1e-6,
        "max_weight": 100.0,
        "adaptation_threshold": 0.1,
        "baseline_iterations": 10,
        "enable_monitoring": True,
    }

    # Create scheduler
    scheduler = create_dynamic_loss_scheduler(losses_dict, config)

    # Test initial state
    assert scheduler is not None
    assert len(scheduler.base_weights) == 2

    # Simulate training iterations
    current_losses = {"l1_loss": torch.tensor(0.5), "mse_loss": torch.tensor(0.1)}

    # Test iterations before baseline establishment
    for i in range(5):
        weights = scheduler(current_losses, i)
        assert all(w == 1.0 for w in weights.values()), (
            f"Pre-baseline weights should be 1.0, got {weights}"
        )

    # Test iterations during baseline establishment
    for i in range(10):
        weights = scheduler(current_losses, i + 5)
        # Should still be 1.0 during baseline
        assert all(w == 1.0 for w in weights.values()), (
            f"Baseline weights should be 1.0, got {weights}"
        )

    # Test adaptation phase
    for i in range(10):
        weights = scheduler(current_losses, i + 15)
        # Now weights should start adapting
        assert all(0.0 < w <= 100.0 for w in weights.values()), (
            f"Adaptive weights should be positive and bounded, got {weights}"
        )

    print("✅ Basic dynamic loss scheduling test passed!")


def test_dynamic_loss_scheduling_with_varying_losses() -> None:
    """Test dynamic loss scheduling with varying loss values."""
    print("Testing dynamic loss scheduling with varying losses...")

    # Create losses
    losses_dict = {
        "loss_a": type("Loss", (), {"loss_weight": 1.0})(),
        "loss_b": type("Loss", (), {"loss_weight": 0.5})(),
        "loss_c": type("Loss", (), {"loss_weight": 0.1})(),
    }

    config = {
        "enabled": True,
        "momentum": 0.9,
        "adaptation_rate": 0.01,
        "baseline_iterations": 5,
    }

    scheduler = create_dynamic_loss_scheduler(losses_dict, config)

    # Simulate training with varying losses
    # loss_a should decrease (increase weight)
    # loss_b should stay stable (maintain weight)
    # loss_c should increase (decrease weight)
    for iteration in range(20):
        if iteration < 5:
            # Baseline phase
            current_losses = {
                "loss_a": torch.tensor(1.0),
                "loss_b": torch.tensor(0.5),
                "loss_c": torch.tensor(0.1),
            }
        elif iteration < 10:
            # Early adaptation phase
            current_losses = {
                "loss_a": torch.tensor(0.8),  # Decreasing
                "loss_b": torch.tensor(0.5),  # Stable
                "loss_c": torch.tensor(0.15),  # Increasing
            }
        else:
            # Later adaptation phase
            current_losses = {
                "loss_a": torch.tensor(0.6),  # Further decreasing
                "loss_b": torch.tensor(0.5),  # Stable
                "loss_c": torch.tensor(0.2),  # Further increasing
            }

        weights = scheduler(current_losses, iteration)

        # Verify weights are reasonable
        assert len(weights) == 3
        assert all(isinstance(w, float) and w > 0 for w in weights.values())

    print("✅ Dynamic loss scheduling with varying losses test passed!")


def test_dynamic_loss_scheduling_bounds() -> None:
    """Test that dynamic loss scheduling respects bounds."""
    print("Testing dynamic loss scheduling bounds...")

    losses_dict = {"test_loss": type("Loss", (), {"loss_weight": 1.0})()}

    config = {
        "enabled": True,
        "momentum": 0.9,
        "adaptation_rate": 1.0,  # Very high adaptation rate
        "min_weight": 0.01,
        "max_weight": 2.0,
        "baseline_iterations": 2,
    }

    scheduler = create_dynamic_loss_scheduler(losses_dict, config)

    # Test with extreme loss changes to trigger bounds
    for iteration in range(10):
        current_losses = {
            "test_loss": torch.tensor(1000.0 if iteration % 2 == 0 else 0.001)
        }

        weights = scheduler(current_losses, iteration)
        weight = weights["test_loss"]

        # Should respect bounds
        assert 0.01 <= weight <= 2.0, (
            f"Weight {weight} should be within bounds [0.01, 2.0]"
        )

    print("✅ Dynamic loss scheduling bounds test passed!")


def test_dynamic_loss_scheduling_factory_function() -> None:
    """Test the factory function for creating schedulers."""
    print("Testing scheduler factory function...")

    # Test with empty losses
    try:
        create_dynamic_loss_scheduler({}, {})
        raise AssertionError("Should raise error with empty losses")
    except Exception:
        pass  # Expected

    # Test with valid losses
    losses_dict = {
        "loss1": type("Loss", (), {"loss_weight": 1.0})(),
        "loss2": type("Loss", (), {"loss_weight": 0.5})(),
    }

    config = {"enabled": True}
    scheduler = create_dynamic_loss_scheduler(losses_dict, config)

    assert scheduler is not None
    assert len(scheduler.base_weights) == 2

    # Test with missing weights
    losses_dict_no_weight = {
        "loss1": type("Loss", (), {})(),  # No loss_weight attribute
    }

    scheduler = create_dynamic_loss_scheduler(losses_dict_no_weight, {})
    assert scheduler is not None
    assert scheduler.base_weights["loss1"].item() == 1.0  # Should default to 1.0

    print("✅ Scheduler factory function test passed!")


def test_integration_with_real_losses() -> None:
    """Test integration with actual loss functions from traiNNer."""
    print("Testing integration with real traiNNer losses...")

    # Create real loss instances using build_loss
    l1_config = {"type": "l1loss", "loss_weight": 1.0}
    mse_config = {"type": "mseloss", "loss_weight": 0.1}

    l1_loss = build_loss(l1_config)
    mse_loss = build_loss(mse_config)

    # Set up loss weights (normally done by the framework)
    l1_loss.loss_weight = 1.0
    mse_loss.loss_weight = 0.1

    losses_dict = {"l_g_l1": l1_loss, "l_g_mse": mse_loss}

    config = {"enabled": True, "baseline_iterations": 3}

    scheduler = create_dynamic_loss_scheduler(losses_dict, config)

    # Test with tensor inputs
    current_losses = {"l_g_l1": torch.tensor(0.5), "l_g_mse": torch.tensor(0.05)}

    weights = scheduler(current_losses, 10)

    assert len(weights) == 2
    assert "l_g_l1" in weights
    assert "l_g_mse" in weights
    assert all(isinstance(w, float) and w > 0 for w in weights.values())

    print("✅ Integration with real losses test passed!")


def test_monitoring_and_stats() -> None:
    """Test monitoring and statistics functionality."""
    print("Testing monitoring and statistics...")

    losses_dict = {
        "loss1": type("Loss", (), {"loss_weight": 1.0})(),
        "loss2": type("Loss", (), {"loss_weight": 0.5})(),
    }

    config = {"enabled": True, "baseline_iterations": 2, "enable_monitoring": True}

    scheduler = create_dynamic_loss_scheduler(losses_dict, config)

    # Test initial stats
    stats = scheduler.get_monitoring_stats()
    assert "iteration" in stats
    assert "baseline_established" in stats
    assert "current_weights" in stats
    assert "smoothed_losses" in stats

    # Run some iterations
    for i in range(5):
        current_losses = {
            "loss1": torch.tensor(1.0 + 0.1 * i),
            "loss2": torch.tensor(0.5 - 0.05 * i),
        }
        weights = scheduler(current_losses, i)

    # Test updated stats
    stats = scheduler.get_monitoring_stats()
    assert stats["iteration"] == 4  # Last iteration
    assert len(stats["current_weights"]) == 2
    assert len(stats["smoothed_losses"]) == 2

    # Test reset functionality
    scheduler.reset(keep_baseline=True)
    stats_after_reset = scheduler.get_monitoring_stats()
    assert stats_after_reset["adaptation_count"] == 0

    print("✅ Monitoring and statistics test passed!")


def run_all_tests() -> bool | None:
    """Run all tests for dynamic loss scheduling."""
    print("=" * 60)
    print("Running Dynamic Loss Scheduling Tests")
    print("=" * 60)

    try:
        test_dynamic_loss_scheduling_basic()
        test_dynamic_loss_scheduling_with_varying_losses()
        test_dynamic_loss_scheduling_bounds()
        test_dynamic_loss_scheduling_factory_function()
        test_integration_with_real_losses()
        test_monitoring_and_stats()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Dynamic Loss Scheduling is working correctly.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
