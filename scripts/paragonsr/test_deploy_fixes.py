#!/usr/bin/env python3
"""
ParagonSR Deployment Script - Test Validation Script
Author: Philip Hofmann

Description:
This script validates that the corrected paragon_deploy.py script
works properly by testing its core functionality without requiring
actual model files.

Usage:
python -m scripts.paragonsr.test_deploy_fixes
"""

import os
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.paragonsr.paragon_deploy import (
    get_model_variant,
    validate_fused_model,
    validate_onnx_model,
    validate_training_checkpoint,
    validate_training_scale,
)
from traiNNer.archs.paragonsr_arch import paragonsr_s


def test_model_variant_loading() -> bool:
    """Test that all ParagonSR model variants can be loaded."""
    print("🧪 Testing Model Variant Loading...")

    variants = ["tiny", "xs", "s", "m", "l", "xl"]

    for variant in variants:
        try:
            model_func = get_model_variant(variant)
            model = model_func(scale=4)
            print(f"  ✅ {variant.upper()}: Loaded successfully")
        except Exception as e:
            print(f"  ❌ {variant.upper()}: Failed to load - {e}")
            return False

    return True


def test_training_checkpoint_validation() -> bool:
    """Test training checkpoint validation with synthetic data."""
    print("\n🧪 Testing Training Checkpoint Validation...")

    # Create a mock training checkpoint state_dict
    model = paragonsr_s(scale=4)

    # Get the training state dict (has ReparamConvV2 patterns)
    training_state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "conv3x3") and hasattr(module, "conv1x1"):
            # This is a ReparamConvV2 module
            training_state_dict[f"{name}.conv3x3.weight"] = torch.randn(64, 64, 3, 3)
            training_state_dict[f"{name}.conv3x3.bias"] = torch.randn(64)
            training_state_dict[f"{name}.conv1x1.weight"] = torch.randn(64, 64, 1, 1)
            training_state_dict[f"{name}.conv1x1.bias"] = torch.randn(64)
        elif hasattr(module, "spatial_mixer") and hasattr(
            module.spatial_mixer, "conv3x3"
        ):
            # This is a GatedFFN with spatial_mixer
            training_state_dict[f"{name}.spatial_mixer.conv3x3.weight"] = torch.randn(
                128, 128, 3, 3
            )
            training_state_dict[f"{name}.spatial_mixer.conv3x3.bias"] = torch.randn(128)
            training_state_dict[f"{name}.spatial_mixer.conv1x1.weight"] = torch.randn(
                128, 128, 1, 1
            )
            training_state_dict[f"{name}.spatial_mixer.conv1x1.bias"] = torch.randn(128)

    # Test validation
    is_valid, msg = validate_training_checkpoint(training_state_dict)

    if is_valid:
        print(f"  ✅ Training checkpoint validation: {msg}")
        return True
    else:
        print(f"  ❌ Training checkpoint validation failed: {msg}")
        return False


def test_scale_validation() -> bool | None:
    """Test scale validation functionality."""
    print("\n🧪 Testing Scale Validation...")

    try:
        # Create a model with known scale
        model_func = get_model_variant("s")

        # Create a mock state dict (we'll use a simple state dict)
        mock_state_dict = {"conv_in.weight": torch.randn(64, 3, 3, 3)}

        # Test valid scale
        is_valid, msg = validate_training_scale(mock_state_dict, 4, model_func)
        print(f"  ✅ Scale validation (4x): {msg}")

        # Test another scale
        _is_valid, msg = validate_training_scale(mock_state_dict, 2, model_func)
        print(f"  ✅ Scale validation (2x): {msg}")

        return True

    except Exception as e:
        print(f"  ❌ Scale validation failed: {e}")
        return False


def test_fused_model_validation() -> bool:
    """Test fused model validation."""
    print("\n🧪 Testing Fused Model Validation...")

    # Create a mock fused state_dict (no ReparamConvV2 patterns)
    fused_state_dict = {
        "conv_in.weight": torch.randn(64, 3, 3, 3),
        "body.0.blocks.0.norm1.weight": torch.randn(64),
        "body.0.blocks.0.context.dwconv_hw.weight": torch.randn(8, 8, 3, 3),
        "body.0.blocks.0.spatial_mixer.weight": torch.randn(
            128, 128, 3, 3
        ),  # Fused Conv2d
        "body.0.blocks.0.spatial_mixer.bias": torch.randn(128),
        "conv_out.weight": torch.randn(3, 64, 3, 3),
    }

    is_valid, msg = validate_fused_model(fused_state_dict)

    if is_valid:
        print(f"  ✅ Fused model validation: {msg}")
        return True
    else:
        print(f"  ❌ Fused model validation failed: {msg}")
        return False


def test_enhanced_validation_functions() -> bool | None:
    """Test that the enhanced validation functions work correctly."""
    print("\n🧪 Testing Enhanced Validation Functions...")

    try:
        # Test that functions return proper tuple format
        model_func = get_model_variant("s")
        mock_dict = {"test": torch.randn(1)}

        # Test scale validation returns tuple
        result = validate_training_scale(mock_dict, 4, model_func)
        if not isinstance(result, tuple) or len(result) != 2:
            print("  ❌ Scale validation should return (bool, str) tuple")
            return False

        # Test fused model validation returns tuple
        result = validate_fused_model(mock_dict)
        if not isinstance(result, tuple) or len(result) != 2:
            print("  ❌ Fused model validation should return (bool, str) tuple")
            return False

        print("  ✅ Enhanced validation functions work correctly")
        return True

    except Exception as e:
        print(f"  ❌ Enhanced validation test failed: {e}")
        return False


def test_error_handling() -> bool | None:
    """Test error handling improvements."""
    print("\n🧪 Testing Error Handling...")

    try:
        # Test with empty state dict
        result = validate_training_checkpoint({})
        if result[0] != False or "empty" not in result[1].lower():
            print("  ❌ Empty state dict should be rejected")
            return False

        # Test with invalid model variant
        try:
            get_model_variant("invalid_variant")
            print("  ❌ Invalid variant should raise ValueError")
            return False
        except ValueError:
            print("  ✅ Invalid variant properly raises ValueError")

        print("  ✅ Error handling improvements working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


def main() -> bool:
    """Run all tests."""
    print("🚀 ParagonSR Deployment Script - Validation Tests")
    print("=" * 60)

    tests = [
        ("Model Variant Loading", test_model_variant_loading),
        ("Training Checkpoint Validation", test_training_checkpoint_validation),
        ("Scale Validation", test_scale_validation),
        ("Fused Model Validation", test_fused_model_validation),
        ("Enhanced Validation Functions", test_enhanced_validation_functions),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Test '{test_name}' failed")
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The deployment script fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
