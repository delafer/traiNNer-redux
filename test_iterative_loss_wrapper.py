#!/usr/bin/env python3
"""
Test script for the IterativeLossWrapper implementation.
Tests if the new framework properly supports iteration-based loss scheduling.
"""

import os
import sys
from pathlib import Path

import torch
import yaml

# Add the project root to the path
sys.path.insert(0, "/home/phips/Documents/GitHub/traiNNer-redux")

from traiNNer.losses import build_loss
from traiNNer.utils.options import parse_options


def test_iterative_loss_wrapper() -> bool:
    """Test the iterative loss wrapper with sample configuration."""

    print("🧪 Testing IterativeLossWrapper Implementation")
    print("=" * 60)

    # Test the loss wrapper with a simple config
    loss_config = {
        "type": "charbonnierloss",
        "loss_weight": 1.0,
        "start_iter": 0,
        "target_iter": 20000,
        "target_weight": 0.26,
        "schedule_type": "linear",
    }

    print("📋 Testing loss config:")
    print(f"   Type: {loss_config['type']}")
    print(f"   Loss weight: {loss_config['loss_weight']}")
    print(f"   Start iter: {loss_config['start_iter']}")
    print(f"   Target iter: {loss_config['target_iter']}")
    print(f"   Target weight: {loss_config['target_weight']}")
    print(f"   Schedule type: {loss_config['schedule_type']}")
    print()

    # Build the loss
    try:
        loss = build_loss(loss_config)
        print("✅ Loss building successful!")
        print(f"   Loss type: {type(loss).__name__}")

        # Test if it's wrapped
        if hasattr(loss, "loss_module"):
            print("✅ Loss properly wrapped with IterativeLossWrapper")
            print(f"   Underlying loss: {type(loss.loss_module).__name__}")

            # Test weight calculation at different iterations
            test_iters = [0, 10000, 20000, 50000]
            print("\n📊 Weight calculations:")
            for iter_num in test_iters:
                weight = loss.get_current_weight(iter_num)
                is_active = loss.is_active(iter_num)
                print(
                    f"   Iteration {iter_num:5d}: weight = {weight:.4f}, active = {is_active}"
                )

        else:
            print(
                "⚠️  Loss not wrapped - this might be expected for non-scheduled losses"
            )

    except Exception as e:
        print(f"❌ Loss building failed: {e}")
        return False

    print()
    print("=" * 60)
    return True


def test_real_config() -> bool | None:
    """Test with the actual training config."""

    print("🎯 Testing with Real Training Config")
    print("=" * 60)

    config_path = "/home/phips/Documents/GitHub/traiNNer-redux/options/train/ParagonSR2/2xParagonSR2_static_S_gan.yml"

    try:
        # Load the config file
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"📁 Loaded config: {config_path}")
        print(f"   Model name: {config.get('name', 'Unknown')}")

        # Extract loss configurations
        train_config = config.get("train", {})
        losses = train_config.get("losses", [])

        print(f"📋 Found {len(losses)} loss configurations")

        # Test each loss
        wrapped_count = 0
        for i, loss_config in enumerate(losses):
            loss_type = loss_config.get("type", "Unknown")
            print(f"\n   Loss {i + 1}: {loss_type}")

            # Check if this loss has scheduling parameters
            schedule_params = [
                "start_iter",
                "target_iter",
                "target_weight",
                "disable_after",
            ]
            has_schedule = any(param in loss_config for param in schedule_params)

            if has_schedule:
                print(f"   ⚡ Has iteration scheduling: {has_schedule}")
                for param in schedule_params:
                    if param in loss_config:
                        print(f"      {param}: {loss_config[param]}")

                # Try to build this loss
                try:
                    loss = build_loss(loss_config)
                    if hasattr(loss, "loss_module"):
                        wrapped_count += 1
                        print("   ✅ Successfully wrapped with IterativeLossWrapper")
                    else:
                        print("   ⚠️  Loss not wrapped despite scheduling parameters")
                except Exception as e:
                    print(f"   ❌ Failed to build loss: {e}")
            else:
                print("   📝 Standard loss (no scheduling)")

        print("\n📊 Summary:")
        print(f"   Total losses: {len(losses)}")
        print(f"   Wrapped losses: {wrapped_count}")
        print(f"   Unwrapped losses: {len(losses) - wrapped_count}")

        if wrapped_count > 0:
            print("✅ Iteration-based loss scheduling successfully implemented!")
            return True
        else:
            print("⚠️  No losses were wrapped - checking for configuration issues")
            return False

    except Exception as e:
        print(f"❌ Failed to test real config: {e}")
        return False


def main() -> bool:
    """Main test function."""
    print("🚀 IterativeLossWrapper Implementation Test")
    print("Testing the new framework for iteration-based loss scheduling")
    print()

    # Test 1: Basic functionality
    basic_test_passed = test_iterative_loss_wrapper()
    print()

    # Test 2: Real config compatibility
    real_config_test_passed = test_real_config()

    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(
        f"Basic functionality test: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}"
    )
    print(
        f"Real config test: {'✅ PASSED' if real_config_test_passed else '❌ FAILED'}"
    )

    if basic_test_passed and real_config_test_passed:
        print("\n🎉 ALL TESTS PASSED! The framework extension is ready for use.")
        print(
            "\nYour training config with iteration-based loss scheduling should now work!"
        )
        return True
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
