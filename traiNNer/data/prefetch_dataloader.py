import logging
import queue
import threading
import time
import warnings
import weakref
from collections.abc import Iterator
from threading import Event, Lock
from typing import Any, Dict, Optional, Set

import torch
from torch.utils.data import DataLoader

from traiNNer.utils import get_root_logger
from traiNNer.utils.redux_options import ReduxOptions


class WorkerHealthMonitor:
    """Monitor worker health and detect dead workers."""

    def __init__(self, dataloader: DataLoader, check_interval: float = 1.0) -> None:
        self.dataloader = weakref.ref(dataloader)
        self.check_interval = check_interval
        self.worker_ids: set[int] = set()
        self.dead_workers: set[int] = set()
        self.last_check = time.time()
        self._lock = Lock()
        self._monitoring = False

    def start_monitoring(self) -> None:
        """Start monitoring worker health."""
        with self._lock:
            if not self._monitoring:
                self._monitoring = True
                self._update_worker_ids()

    def stop_monitoring(self) -> None:
        """Stop monitoring worker health."""
        with self._lock:
            self._monitoring = False

    def _update_worker_ids(self) -> None:
        """Update the set of active worker IDs."""
        try:
            dl = self.dataloader()
            if dl is not None and hasattr(dl, "_workers"):
                with self._lock:
                    current_workers = {w.pid for w in dl._workers if w.is_alive()}
                    self.worker_ids = current_workers

                    # Update dead workers
                    new_dead = self.worker_ids - current_workers
                    if new_dead:
                        self.dead_workers.update(new_dead)
                        logger = get_root_logger()
                        logger.warning(f"Detected dead workers: {new_dead}")

        except (AttributeError, RuntimeError):
            # DataLoader might be shutting down or cleaned up
            pass

    def is_worker_alive(self, worker_id: int | None) -> bool:
        """Check if a specific worker is alive."""
        if worker_id is None:
            return True

        with self._lock:
            return worker_id in self.worker_ids and worker_id not in self.dead_workers

    def check_health(self) -> bool:
        """Perform health check on workers."""
        if not self._monitoring:
            return True

        current_time = time.time()
        if current_time - self.last_check >= self.check_interval:
            self._update_worker_ids()
            self.last_check = current_time

        with self._lock:
            return len(self.dead_workers) == 0

    def cleanup_dead_workers(self) -> None:
        """Clean up references to dead workers."""
        with self._lock:
            self.dead_workers.clear()


class RobustPrefetchGenerator(threading.Thread):
    """Enhanced prefetch generator with robust worker management and timeout protection."""

    def __init__(self, generator: DataLoader, num_prefetch_queue: int = 3) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.running = True
        self.prefetch_timeout = 30.0  # 30 seconds timeout for prefetch operations
        self.health_monitor = WorkerHealthMonitor(generator)
        self.fallback_mode = False
        self.error_count = 0
        self.max_errors = 5
        self._shutdown_event = Event()

        # Start monitoring worker health
        self.health_monitor.start_monitoring()
        self.start()

        logger = get_root_logger()
        logger.info("RobustPrefetchGenerator initialized with worker health monitoring")

    def run(self) -> None:
        """Main prefetch loop with error handling and timeout protection."""
        try:
            while self.running and not self._shutdown_event.is_set():
                if self.fallback_mode:
                    break

                # Check worker health before each prefetch attempt
                if not self.health_monitor.check_health():
                    logger = get_root_logger()
                    logger.warning("Worker health check failed, attempting cleanup")
                    self.health_monitor.cleanup_dead_workers()
                    self.error_count += 1

                    if self.error_count >= self.max_errors:
                        logger.error(
                            "Too many worker health failures, entering fallback mode"
                        )
                        self.fallback_mode = True
                        break

                    # Wait a bit before retrying
                    time.sleep(0.1)
                    continue

                try:
                    # Use timeout for generator iteration
                    item = self._safe_next_with_timeout()

                    if item is None:
                        # Generator exhausted
                        self.queue.put(None)
                        break

                    # Reset error count on successful prefetch
                    self.error_count = 0
                    self.queue.put(item)

                except (KeyError, RuntimeError, AttributeError) as e:
                    # Handle worker-related errors (like KeyError: 3)
                    logger = get_root_logger()
                    logger.warning(f"Worker error detected: {type(e).__name__}: {e}")

                    self.error_count += 1
                    self.health_monitor._update_worker_ids()

                    if self.error_count >= self.max_errors:
                        logger.error(
                            "Maximum worker errors reached, entering fallback mode"
                        )
                        self.fallback_mode = True
                        break

                    # Brief pause before retrying
                    time.sleep(0.05)

                except Exception as e:
                    # Handle other unexpected errors
                    logger = get_root_logger()
                    logger.error(
                        f"Unexpected error in prefetch generator: {type(e).__name__}: {e}"
                    )
                    self.error_count += 1

                    if self.error_count >= self.max_errors:
                        self.fallback_mode = True
                        break

                    time.sleep(0.1)

        except Exception as e:
            logger = get_root_logger()
            logger.error(
                f"Critical error in prefetch generator: {type(e).__name__}: {e}"
            )

        finally:
            # Clean shutdown
            self.health_monitor.stop_monitoring()
            self.running = False

            # Put None to signal completion even if there were errors
            try:
                self.queue.put(None, timeout=0.1)
            except queue.Full:
                pass

    def _safe_next_with_timeout(self) -> Any:
        """Get next item with timeout protection."""
        start_time = time.time()

        while time.time() - start_time < self.prefetch_timeout:
            try:
                # Check if we've been shut down
                if self._shutdown_event.is_set():
                    return None

                # Try to get next item with a short timeout
                if hasattr(self.generator, "_worker_manager"):
                    # PyTorch DataLoader with worker manager
                    item = next(self.generator)
                else:
                    # Standard iterator
                    item = next(self.generator)

                return item

            except (KeyError, RuntimeError, AttributeError) as e:
                # Worker-related errors that might be temporary
                logger = get_root_logger()
                logger.warning(f"Worker access error: {type(e).__name__}: {e}")

                # Check if this is the specific KeyError: 3 issue
                if "KeyError" in str(e) or "task_info" in str(e):
                    self.health_monitor._update_worker_ids()

                raise

            except StopIteration:
                return None

            except Exception as e:
                # Other exceptions - might be temporary
                if time.time() - start_time < self.prefetch_timeout:
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                else:
                    raise

        raise TimeoutError(
            f"Prefetch operation timed out after {self.prefetch_timeout} seconds"
        )

    def __next__(self) -> Any:
        """Get next item from the prefetch queue."""
        try:
            next_item = self.queue.get(timeout=1.0)
            if next_item is None:
                raise StopIteration
            return next_item
        except queue.Empty:
            if not self.running or self.fallback_mode:
                raise StopIteration
            raise

    def __iter__(self) -> Iterator:
        return self

    def stop(self) -> None:
        """Stop the prefetch generator gracefully."""
        self.running = False
        self._shutdown_event.set()
        self.health_monitor.stop_monitoring()

        # Wait for thread to finish (with timeout)
        if self.is_alive():
            self.join(timeout=2.0)


class PrefetchDataLoader(DataLoader):
    """Enhanced prefetch dataloader with robust worker management and fallback capabilities."""

    def __init__(
        self, num_prefetch_queue: int = 3, timeout: float = 30.0, **kwargs
    ) -> None:
        self.num_prefetch_queue = num_prefetch_queue
        self.prefetch_timeout = timeout
        self.fallback_dataloader = None
        self._using_fallback = False
        super().__init__(**kwargs)

    def __iter__(self) -> RobustPrefetchGenerator:  # type: ignore
        # Create fallback dataloader for graceful degradation
        self.fallback_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.prefetch_timeout,
            worker_init_fn=self.worker_init_fn,
        )

        try:
            generator = RobustPrefetchGenerator(
                super().__iter__(), self.num_prefetch_queue
            )
            return generator
        except Exception as e:
            logger = get_root_logger()
            logger.warning(
                f"Failed to create robust prefetch generator: {e}. Falling back to normal dataloader."
            )
            self._using_fallback = True
            return RobustPrefetchGenerator(
                iter(self.fallback_dataloader), self.num_prefetch_queue
            )

    def reset(self) -> None:
        """Reset the dataloader state."""
        if (
            hasattr(self, "_using_fallback")
            and self._using_fallback
            and self.fallback_dataloader
        ):
            # Reset fallback dataloader
            self.fallback_dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                timeout=self.prefetch_timeout,
                worker_init_fn=self.worker_init_fn,
            )


class CPUPrefetcher:
    """Enhanced CPU prefetcher with error handling."""

    def __init__(self, loader: DataLoader, timeout: float = 30.0) -> None:
        self.ori_loader = loader
        self.loader = iter(loader)
        self.prefetch_timeout = timeout
        self.error_count = 0
        self.max_errors = 3

    def next(self) -> Any:
        """Get next batch with timeout and error handling."""
        try:
            # Set a timeout for the next() call
            start_time = time.time()
            result = next(self.loader)

            # Reset error count on success
            self.error_count = 0
            return result

        except StopIteration:
            return None

        except (KeyError, RuntimeError, AttributeError) as e:
            logger = get_root_logger()
            logger.warning(f"Worker error in CPU prefetcher: {type(e).__name__}: {e}")

            self.error_count += 1
            if self.error_count >= self.max_errors:
                logger.error("Too many worker errors in CPU prefetcher, resetting")
                self.loader = iter(self.ori_loader)
                self.error_count = 0

            raise

    def reset(self) -> None:
        """Reset the prefetcher."""
        self.loader = iter(self.ori_loader)
        self.error_count = 0


class CUDAPrefetcher:
    """Enhanced CUDA prefetcher with robust error handling."""

    def __init__(self, loader: DataLoader, opt: ReduxOptions) -> None:
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device("cuda" if opt.num_gpu != 0 else "cpu")
        self.batch = None
        self.error_count = 0
        self.max_errors = 3

        # Check if pin_memory is enabled to avoid compatibility issues with bf16 + channels_last
        pin_memory_enabled = getattr(loader, "pin_memory", False)
        use_channels_last = (
            opt.use_amp and opt.use_channels_last and not pin_memory_enabled
        )

        if opt.use_amp and opt.use_channels_last and pin_memory_enabled:
            logger = get_root_logger()
            logger.warning(
                "Auto-disable channels_last memory format due to pin_memory incompatibility with bf16. "
                "For better performance, consider setting 'pin_memory: false' in your dataset configuration."
            )

        self.memory_format = (
            torch.channels_last if use_channels_last else torch.preserve_format
        )
        self.preload()

    def preload(self) -> None:
        """Preload next batch with error handling."""
        try:
            self.batch = next(self.loader)  # self.batch is a dict
            self.error_count = 0  # Reset error count on success
        except StopIteration:
            self.batch = None
            return None
        except (KeyError, RuntimeError, AttributeError) as e:
            logger = get_root_logger()
            logger.warning(
                f"Worker error in CUDA prefetcher preload: {type(e).__name__}: {e}"
            )

            self.error_count += 1
            if self.error_count >= self.max_errors:
                logger.error("Too many worker errors in CUDA prefetcher, resetting")
                self.loader = iter(self.ori_loader)
                try:
                    self.batch = next(self.loader)
                except StopIteration:
                    self.batch = None
                self.error_count = 0
            else:
                # Brief pause before retry
                time.sleep(0.01)
                raise

        # put tensors to gpu
        if self.batch is not None:
            with torch.cuda.stream(self.stream):  # type: ignore
                for k, v in self.batch.items():
                    if torch.is_tensor(v):
                        self.batch[k] = self.batch[k].to(
                            device=self.device,
                            memory_format=self.memory_format,
                            non_blocking=True,
                        )

    def next(self) -> Any:
        """Get next batch with stream synchronization."""
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self) -> None:
        """Reset the prefetcher."""
        self.loader = iter(self.ori_loader)
        self.error_count = 0
        self.preload()
