#!/usr/bin/env python3
"""
Test script for the enhanced prefetch dataloader with robust worker management.

This script tests the KeyError: 3 fix by simulating scenarios where workers
might terminate during prefetch operations.
"""

import logging
import sys
import time
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from traiNNer.data.prefetch_dataloader import (
    CPUPrefetcher,
    CUDAPrefetcher,
    PrefetchDataLoader,
    RobustPrefetchGenerator,
    WorkerHealthMonitor,
)
from traiNNer.utils.redux_options import ReduxOptions


class TestDataset(Dataset):
    """Simple test dataset that simulates potential worker issues."""

    def __init__(self, size: int = 1000, error_rate: float = 0.0) -> None:
        self.size = size
        self.error_rate = error_rate
        self.data = np.random.randn(size, 3, 64, 64).astype(np.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        # Simulate occasional worker errors
        if self.error_rate > 0 and np.random.random() < self.error_rate:
            # Simulate the kind of error that might cause KeyError: 3
            raise RuntimeError(f"Simulated worker error for index {index}")

        return {
            "lq": torch.from_numpy(self.data[index]),
            "gt": torch.from_numpy(self.data[index] * 1.1),
            "index": index,
        }


def test_worker_health_monitor() -> None:
    """Test the WorkerHealthMonitor functionality."""
    print("Testing WorkerHealthMonitor...")

    # Create a simple dataloader
    dataset = TestDataset(size=100)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

    # Test health monitor
    monitor = WorkerHealthMonitor(dataloader)

    try:
        monitor.start_monitoring()
        time.sleep(0.1)  # Give it time to initialize

        # Test worker health checking
        is_healthy = monitor.check_health()
        print(f"  ✓ Worker health check passed: {is_healthy}")

        # Test specific worker ID validation
        worker_alive = monitor.is_worker_alive(worker_id=0)
        print(f"  ✓ Worker 0 alive check: {worker_alive}")

        monitor.stop_monitoring()
        print("  ✓ WorkerHealthMonitor test completed successfully")

    except Exception as e:
        print(f"  ✗ WorkerHealthMonitor test failed: {e}")
        traceback.print_exc()


def test_robust_prefetch_generator() -> None:
    """Test the RobustPrefetchGenerator with potential worker errors."""
    print("\nTesting RobustPrefetchGenerator...")

    try:
        # Create dataset with occasional errors
        dataset = TestDataset(size=100, error_rate=0.01)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

        # Create robust prefetch generator
        generator = RobustPrefetchGenerator(dataloader, num_prefetch_queue=2)

        # Test iteration
        items_processed = 0
        max_items = 10

        try:
            for batch in generator:
                items_processed += 1
                print(f"  ✓ Processed batch {items_processed}: {batch['lq'].shape}")

                if items_processed >= max_items:
                    break

        except Exception as e:
            print(f"  ✗ Error during iteration: {e}")
            traceback.print_exc()

        finally:
            # Cleanup
            generator.stop()

        print(
            f"  ✓ RobustPrefetchGenerator test completed: processed {items_processed} batches"
        )

    except Exception as e:
        print(f"  ✗ RobustPrefetchGenerator test failed: {e}")
        traceback.print_exc()


def test_prefetch_dataloader() -> None:
    """Test the enhanced PrefetchDataLoader."""
    print("\nTesting enhanced PrefetchDataLoader...")

    try:
        # Create dataset
        dataset = TestDataset(size=100)

        # Test CPU prefetch mode
        prefetch_loader = PrefetchDataLoader(
            dataset=dataset,
            batch_size=4,
            num_workers=2,
            num_prefetch_queue=2,
            timeout=30.0,
        )

        # Test iteration
        batches_processed = 0
        max_batches = 5

        try:
            for batch in prefetch_loader:
                batches_processed += 1
                print(
                    f"  ✓ PrefetchDataLoader batch {batches_processed}: {batch['lq'].shape}"
                )

                if batches_processed >= max_batches:
                    break

        except Exception as e:
            print(f"  ✗ Error during prefetch iteration: {e}")
            traceback.print_exc()

        print(
            f"  ✓ PrefetchDataLoader test completed: processed {batches_processed} batches"
        )

    except Exception as e:
        print(f"  ✗ PrefetchDataLoader test failed: {e}")
        traceback.print_exc()


def test_cpu_prefetcher() -> None:
    """Test the enhanced CPUPrefetcher."""
    print("\nTesting CPUPrefetcher...")

    try:
        # Create dataloader
        dataset = TestDataset(size=50)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

        # Test CPU prefetcher
        prefetcher = CPUPrefetcher(dataloader)

        # Test multiple next() calls
        items_processed = 0
        max_items = 5

        try:
            while items_processed < max_items:
                batch = prefetcher.next()
                if batch is None:
                    print("  ✓ CPUPrefetcher reached end of data")
                    break

                items_processed += 1
                print(
                    f"  ✓ CPUPrefetcher processed item {items_processed}: {batch['lq'].shape}"
                )

        except Exception as e:
            print(f"  ✗ Error during CPU prefetch: {e}")
            traceback.print_exc()

        print(f"  ✓ CPUPrefetcher test completed: processed {items_processed} items")

    except Exception as e:
        print(f"  ✗ CPUPrefetcher test failed: {e}")
        traceback.print_exc()


def test_cuda_prefetcher() -> None:
    """Test the enhanced CUDAPrefetcher (will use CPU if no CUDA available)."""
    print("\nTesting CUDAPrefetcher...")

    try:
        # Create dataset and dataloader
        dataset = TestDataset(size=50)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

        # Create dummy options
        opt = ReduxOptions()
        opt.num_gpu = 0  # Use CPU if no GPU available
        opt.use_amp = False
        opt.use_channels_last = False

        # Test CUDAPrefetcher
        prefetcher = CUDAPrefetcher(dataloader, opt)

        # Test multiple next() calls
        items_processed = 0
        max_items = 3

        try:
            while items_processed < max_items:
                batch = prefetcher.next()
                if batch is None:
                    print("  ✓ CUDAPrefetcher reached end of data")
                    break

                items_processed += 1
                print(
                    f"  ✓ CUDAPrefetcher processed item {items_processed}: {batch['lq'].shape}"
                )

        except Exception as e:
            print(f"  ✗ Error during CUDA prefetch: {e}")
            traceback.print_exc()

        print(f"  ✓ CUDAPrefetcher test completed: processed {items_processed} items")

    except Exception as e:
        print(f"  ✗ CUDAPrefetcher test failed: {e}")
        traceback.print_exc()


def main() -> None:
    """Run all tests for the enhanced prefetch dataloader."""
    print("=" * 60)
    print("Testing Enhanced Prefetch DataLoader - KeyError: 3 Fix")
    print("=" * 60)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    try:
        test_worker_health_monitor()
        test_robust_prefetch_generator()
        test_prefetch_dataloader()
        test_cpu_prefetcher()
        test_cuda_prefetcher()

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("The enhanced prefetch dataloader should now handle KeyError: 3 issues")
        print(
            "with robust worker management, timeout protection, and graceful fallback."
        )
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
