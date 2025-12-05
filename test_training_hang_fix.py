#!/usr/bin/env python3
"""
Test script to verify the training hang fix is working.

This script tests the simplified prefetch dataloader and can be used
to verify that training can proceed past the initialization phase.
"""

import sys
import time
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Dataset
from traiNNer.data.prefetch_dataloader import CPUPrefetcher, PrefetchDataLoader


class SimpleTestDataset(Dataset):
    """Simple test dataset for testing dataloader functionality."""

    def __init__(self, size: int = 100) -> None:
        self.size = size
        self.data = torch.randn(size, 3, 64, 64)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {
            "lq": self.data[index],
            "gt": self.data[index] * 1.1,
            "index": index,
        }


def test_simplified_prefetch_dataloader() -> bool | None:
    """Test the simplified prefetch dataloader."""
    print("Testing simplified PrefetchDataLoader...")

    try:
        # Create test dataset and dataloader
        dataset = SimpleTestDataset(size=50)

        # Test with reduced workers to avoid issues
        prefetch_loader = PrefetchDataLoader(
            dataset=dataset,
            batch_size=4,
            num_workers=2,
            num_prefetch_queue=2,
            timeout=10.0,
        )

        print("  ✓ PrefetchDataLoader created successfully")
        print(f"  ✓ Dataset size: {len(dataset)}")
        print(f"  ✓ Batch size: {prefetch_loader.batch_size}")
        print(f"  ✓ Num workers: {prefetch_loader.num_workers}")

        # Test iteration
        batches_processed = 0
        max_batches = 3

        print("  Testing iteration...")
        for batch in prefetch_loader:
            batches_processed += 1
            print(f"    ✓ Batch {batches_processed}: lq shape {batch['lq'].shape}")

            if batches_processed >= max_batches:
                break

        print(f"  ✓ Successfully processed {batches_processed} batches")
        return True

    except Exception as e:
        print(f"  ✗ PrefetchDataLoader test failed: {e}")
        traceback.print_exc()
        return False


def test_cpu_prefetcher() -> bool | None:
    """Test the simplified CPU prefetcher."""
    print("\nTesting simplified CPUPrefetcher...")

    try:
        # Create simple dataloader
        dataset = SimpleTestDataset(size=30)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

        # Test CPU prefetcher
        prefetcher = CPUPrefetcher(dataloader, timeout=10.0)

        print("  ✓ CPUPrefetcher created successfully")

        # Test multiple next() calls
        items_processed = 0
        max_items = 3

        print("  Testing prefetch iteration...")
        while items_processed < max_items:
            batch = prefetcher.next()
            if batch is None:
                print("    ✓ Reached end of data")
                break

            items_processed += 1
            print(f"    ✓ Item {items_processed}: lq shape {batch['lq'].shape}")

        print(f"  ✓ Successfully processed {items_processed} items")
        return True

    except Exception as e:
        print(f"  ✗ CPUPrefetcher test failed: {e}")
        traceback.print_exc()
        return False


def test_training_compatibility() -> bool | None:
    """Test compatibility with training setup."""
    print("\nTesting training compatibility...")

    try:
        # Simulate training setup
        dataset = SimpleTestDataset(size=100)

        # Test different configurations that might be used in training
        configs = [
            {"batch_size": 8, "num_workers": 4, "prefetch_mode": "cpu"},
            {"batch_size": 4, "num_workers": 2, "prefetch_mode": "cpu"},
            {"batch_size": 16, "num_workers": 1, "prefetch_mode": "cpu"},
        ]

        for i, config in enumerate(configs):
            print(f"  Testing config {i + 1}: {config}")

            # Create prefetch dataloader
            prefetch_loader = PrefetchDataLoader(
                dataset=dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                timeout=10.0,
            )

            # Test a few iterations
            batches = 0
            for _batch in prefetch_loader:
                batches += 1
                if batches >= 2:  # Just test a couple batches
                    break

            print(f"    ✓ Config {i + 1} passed: {batches} batches processed")

        print("  ✓ All training configurations tested successfully")
        return True

    except Exception as e:
        print(f"  ✗ Training compatibility test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests for the training hang fix."""
    print("=" * 60)
    print("Testing Training Hang Fix - Simplified Prefetch DataLoader")
    print("=" * 60)

    try:
        success = True

        # Run all tests
        success &= test_simplified_prefetch_dataloader()
        success &= test_cpu_prefetcher()
        success &= test_training_compatibility()

        print("\n" + "=" * 60)
        if success:
            print("✅ ALL TESTS PASSED!")
            print("The simplified prefetch dataloader should resolve training hangs.")
            print("Training should now proceed past the initialization phase.")
        else:
            print("❌ SOME TESTS FAILED!")
            print("There may still be issues with the prefetch dataloader.")
        print("=" * 60)

        return success

    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
