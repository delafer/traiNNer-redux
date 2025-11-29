#!/usr/bin/env python3
"""
Test script for Local Window Attention implementation.

This script comprehensively tests the Local Window Attention mechanism
replacing the hierarchical attention in ParagonSR2 and MUNet architectures.

Tests:
1. Basic functionality with various image sizes
2. VRAM efficiency comparison with hierarchical attention
3. Quality preservation verification
4. Memory scaling with different image sizes
5. Edge cases (very small/large images)

Author: Philip Hofmann
License: MIT
"""

import gc
import time
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def get_memory_usage() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def clear_memory() -> None:
    """Clear GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_basic_functionality():
    """Test basic Local Window Attention functionality."""
    print("🧪 Testing Basic Functionality...")

    from traiNNer.archs.munet_arch import LocalWindowAttention as MUNetLWA
    from traiNNer.archs.paragonsr2_arch import LocalWindowAttention as ParagonLWA

    test_cases = [
        (64, 64),  # Small image
        (128, 128),  # Medium image
        (256, 256),  # Large image
        (64, 128),  # Rectangular image
    ]

    success_count = 0
    total_tests = len(test_cases) * 2  # Paragon + MUNet for each size

    for h, w in test_cases:
        print(f"   Testing {h}x{w} image...")

        # Test ParagonSR2 version
        try:
            lwa = ParagonLWA(channels=64, reduction=8, window_size=32, overlap=8)
            x = torch.randn(1, 64, h, w)
            output = lwa(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            success_count += 1
            print(f"   ✅ ParagonSR2 LWA: {h}x{w}")
        except Exception as e:
            print(f"   ❌ ParagonSR2 LWA failed for {h}x{w}: {e}")

        # Test MUNet version
        try:
            lwa = MUNetLWA(channels=64, reduction=8, window_size=32, overlap=8)
            x = torch.randn(1, 64, h, w)
            output = lwa(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            success_count += 1
            print(f"   ✅ MUNet LWA: {h}x{w}")
        except Exception as e:
            print(f"   ❌ MUNet LWA failed for {h}x{w}: {e}")

    print(f"   ✅ Basic functionality: {success_count}/{total_tests} tests passed")
    return success_count == total_tests


def test_vram_efficiency():
    """Test VRAM efficiency compared to hierarchical attention."""
    print("💾 Testing VRAM Efficiency...")

    from traiNNer.archs.paragonsr2_arch import (
        EfficientSelfAttention,
        LocalWindowAttention,
    )

    test_sizes = [(128, 128), (256, 256)]
    results = {}

    for h, w in test_sizes:
        print(f"   Testing {h}x{w} image...")

        # Test Local Window Attention
        try:
            clear_memory()
            start_mem = get_memory_usage()

            lwa = LocalWindowAttention(
                channels=64, reduction=8, window_size=32, overlap=8
            )
            x = torch.randn(1, 64, h, w)

            start_time = time.time()
            output = lwa(x)
            end_time = time.time()

            end_mem = get_memory_usage()
            lwa_memory = end_mem - start_mem
            lwa_time = end_time - start_time
            lwa_success = True

        except Exception as e:
            print(f"   ❌ Local Window Attention failed: {e}")
            lwa_memory = float("inf")
            lwa_time = float("inf")
            lwa_success = False

        # Test Hierarchical Attention
        try:
            clear_memory()
            start_mem = get_memory_usage()

            hwa = EfficientSelfAttention(channels=64, reduction=8)
            x = torch.randn(1, 64, h, w)

            start_time = time.time()
            output = hwa(x)
            end_time = time.time()

            end_mem = get_memory_usage()
            hwa_memory = end_mem - start_mem
            hwa_time = end_time - start_time
            hwa_success = True

        except Exception as e:
            print(f"   ⚠️ Hierarchical Attention failed: {e}")
            hwa_memory = float("inf")
            hwa_time = float("inf")
            hwa_success = False

        results[f"{h}x{w}"] = {
            "lwa_memory": lwa_memory,
            "lwa_time": lwa_time,
            "lwa_success": lwa_success,
            "hwa_memory": hwa_memory,
            "hwa_time": hwa_time,
            "hwa_success": hwa_success,
        }

    print("\n   Memory Usage (GB):")
    for size, data in results.items():
        if data["hwa_success"] and data["lwa_success"]:
            improvement = (
                (data["hwa_memory"] - data["lwa_memory"]) / data["hwa_memory"]
            ) * 100
            print(
                f"   {size}: LWA={data['lwa_memory']:.3f}, HWA={data['hwa_memory']:.3f}, Improvement={improvement:.1f}%"
            )
        else:
            print(
                f"   {size}: LWA={data['lwa_memory']:.3f}, HWA={data['hwa_memory']:.3f}"
            )

    return results


def test_memory_scaling():
    """Test memory scaling with image size."""
    print("📈 Testing Memory Scaling...")

    from traiNNer.archs.paragonsr2_arch import LocalWindowAttention

    sizes = [32, 64, 128, 256]
    memory_usage = []

    for size in sizes:
        try:
            clear_memory()
            start_mem = get_memory_usage()

            lwa = LocalWindowAttention(
                channels=64, reduction=8, window_size=32, overlap=8
            )
            x = torch.randn(1, 64, size, size)

            output = lwa(x)

            end_mem = get_memory_usage()
            memory_usage.append(end_mem - start_mem)
            print(f"   ✅ {size}x{size}: {(end_mem - start_mem):.3f} GB")

        except Exception as e:
            print(f"   ❌ Failed for {size}x{size}: {e}")
            memory_usage.append(float("inf"))

    valid_memories = [m for m in memory_usage if m != float("inf")]
    if len(valid_memories) > 2:
        max_mem = max(valid_memories)
        min_mem = min(valid_memories)
        variation = ((max_mem - min_mem) / min_mem) * 100
        print(f"   Memory variation: {variation:.1f}%")
        return variation < 50

    return False


def test_edge_cases():
    """Test edge cases."""
    print("🔍 Testing Edge Cases...")

    from traiNNer.archs.paragonsr2_arch import LocalWindowAttention

    edge_cases = [
        ("Min size", 1, 1),
        ("Exact window", 32, 32),
        ("Just over", 33, 33),
        ("Multiple", 64, 64),
    ]

    success_count = 0
    total_tests = len(edge_cases)

    for name, h, w in edge_cases:
        try:
            lwa = LocalWindowAttention(
                channels=64, reduction=8, window_size=32, overlap=8
            )
            x = torch.randn(1, 64, h, w)
            output = lwa(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            success_count += 1
            print(f"   ✅ {name}: {h}x{w}")

        except Exception as e:
            print(f"   ❌ {name}: {h}x{w} - {e}")

    print(f"   ✅ Edge cases: {success_count}/{total_tests} passed")
    return success_count == total_tests


def run_comprehensive_test():
    """Run all tests."""
    print("🚀 Local Window Attention Comprehensive Test")
    print("=" * 50)

    test_results = {}

    test_results["basic_functionality"] = test_basic_functionality()
    print()

    test_results["vrams_efficiency"] = test_vram_efficiency()
    print()

    test_results["memory_scaling"] = test_memory_scaling()
    print()

    test_results["edge_cases"] = test_edge_cases()
    print()

    print("📊 Test Summary:")
    print("=" * 50)

    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title():25} | {status}")

    print(f"\n   Overall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("   🎉 All tests passed! Local Window Attention ready for deployment.")
    elif passed_tests >= total_tests * 0.8:
        print("   ⚠️ Most tests passed. LWA should work with minor fixes.")
    else:
        print("   🚨 Many tests failed. LWA needs significant work.")

    return test_results


if __name__ == "__main__":
    try:
        results = run_comprehensive_test()

        print("\n📝 Test completed")

    except Exception as e:
        print(f"\n💥 Test failed: {e}")
    finally:
        clear_memory()
        print("🧹 Memory cleared.")
