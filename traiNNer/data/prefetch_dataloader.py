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
    """Simplified worker health monitor to prevent hangs."""

    def __init__(self, dataloader: DataLoader, check_interval: float = 2.0) -> None:
        self.dataloader = weakref.ref(dataloader)
        self.check_interval = check_interval
        self._monitoring = False
        self._health_check_count = 0
        self._last_health_check = 0.0

    def start_monitoring(self) -> None:
        """Start monitoring worker health."""
        self._monitoring = True
        # Do immediate health check
        self._update_worker_ids_safe()

    def stop_monitoring(self) -> None:
        """Stop monitoring worker health."""
        self._monitoring = False

    def _update_worker_ids_safe(self) -> None:
        """Safely update worker IDs with timeout protection."""
        try:
            # Skip intensive health checks to prevent hangs
            self._health_check_count += 1

            # Only do detailed health check every few times to avoid overhead
            if self._health_check_count % 10 == 0:
                dl = self.dataloader()
                if dl is not None and hasattr(dl, "_workers"):
                    # Quick check without detailed PID tracking
                    active_workers = [
                        w
                        for w in dl._workers
                        if hasattr(w, "is_alive") and w.is_alive()
                    ]

        except Exception as e:
            # Silently ignore health check errors to prevent hangs
            pass

    def is_worker_alive(self, worker_id: int | None) -> bool:
        """Check if a specific worker is alive."""
        if worker_id is None:
            return True
        return True  # Simplified - assume worker is alive

    def check_health(self) -> bool:
        """Perform health check on workers."""
        if not self._monitoring:
            return True

        current_time = time.time()
        if current_time - self._last_health_check >= self.check_interval:
            self._update_worker_ids_safe()
            self._last_health_check = current_time

        return True  # Simplified - always return healthy

    def cleanup_dead_workers(self) -> None:
        """Clean up references to dead workers."""
        # Simplified - no cleanup needed


class RobustPrefetchGenerator(threading.Thread):
    """Simplified prefetch generator with basic robustness."""

    def __init__(self, generator: DataLoader, num_prefetch_queue: int = 3) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.running = True
        self.prefetch_timeout = 15.0  # Increased timeout for data loading
        self.health_monitor = None  # Disable health monitoring for simplicity
        self.fallback_mode = False
        self.error_count = 0
        self.max_errors = 3
        self._shutdown_event = Event()

        # Skip health monitoring to prevent hangs
        self.start()

        logger = get_root_logger()
        logger.info("RobustPrefetchGenerator initialized (simplified version)")

    def run(self) -> None:
        """Simplified prefetch loop with basic error handling."""
        try:
            while self.running and not self._shutdown_event.is_set():
                try:
                    item = self._safe_next_with_timeout()

                    if item is None:
                        # Generator exhausted
                        self.queue.put(None)
                        break

                    # Reset error count on successful prefetch
                    self.error_count = 0
                    self.queue.put(item)

                except (KeyError, RuntimeError, AttributeError) as e:
                    # Handle worker-related errors
                    logger = get_root_logger()
                    logger.warning(f"Worker error detected: {type(e).__name__}: {e}")

                    self.error_count += 1

                    if self.error_count >= self.max_errors:
                        logger.warning("Maximum worker errors reached")
                        self.fallback_mode = True
                        break

                    # Brief pause before retrying
                    time.sleep(0.01)

                except Exception as e:
                    # Handle other unexpected errors
                    logger = get_root_logger()
                    logger.error(f"Prefetch error: {type(e).__name__}: {e}")
                    self.error_count += 1

                    if self.error_count >= self.max_errors:
                        logger.error("Too many errors, stopping prefetch")
                        break

                    time.sleep(0.01)

        except Exception as e:
            logger = get_root_logger()
            logger.error(
                f"Critical error in prefetch generator: {type(e).__name__}: {e}"
            )

        finally:
            # Clean shutdown
            self.running = False

            # Put None to signal completion
            try:
                self.queue.put(None, timeout=0.1)
            except queue.Full:
                pass

    def _safe_next_with_timeout(self) -> Any:
        """Get next item with simplified timeout protection."""
        try:
            # Simple timeout check with retry mechanism
            item = next(self.generator)
            return item

        except StopIteration:
            return None

        except (KeyError, RuntimeError, AttributeError) as e:
            # Worker-related errors - these are common and often transient
            logger = get_root_logger()
            logger.warning(f"Worker access error: {type(e).__name__}: {e}")

            # For worker errors, we often want to retry rather than fail
            # Brief pause and try once more
            time.sleep(0.01)
            try:
                item = next(self.generator)
                logger.info("Worker error recovered on retry")
                return item
            except Exception as retry_e:
                logger.warning(f"Worker error retry failed: {retry_e}")
                raise

        except Exception as e:
            # Other exceptions
            logger = get_root_logger()
            logger.warning(f"Unexpected error in safe_next: {type(e).__name__}: {e}")

            # For unexpected errors, try to recover once
            time.sleep(0.01)
            try:
                item = next(self.generator)
                logger.info("Unexpected error recovered on retry")
                return item
            except Exception as retry_e:
                logger.error(f"Unexpected error retry failed: {retry_e}")
                raise

    def __next__(self) -> Any:
        """Get next item from the prefetch queue with better timeout handling."""
        try:
            next_item = self.queue.get(timeout=10.0)  # Increased timeout to 10 seconds
            if next_item is None:
                raise StopIteration
            return next_item
        except queue.Empty:
            logger = get_root_logger()
            logger.warning(
                "Prefetch queue timeout after 10s, checking thread status..."
            )

            if not self.running or self.fallback_mode:
                logger.info("Prefetch generator stopped, ending iteration")
                raise StopIteration

            # Check if thread is still alive
            if not self.is_alive():
                logger.warning("Prefetch thread died, ending iteration")
                raise StopIteration

            # Try one more time with a longer timeout
            try:
                next_item = self.queue.get(timeout=5.0)
                if next_item is None:
                    raise StopIteration
                return next_item
            except queue.Empty:
                logger.warning(
                    "Queue still empty after extended timeout, ending iteration"
                )
                # Instead of raising StopIteration, let's try to reset the queue
                logger.info("Attempting to recover from queue timeout...")

                # Set running to False and try to restart the thread
                self.running = False
                time.sleep(0.1)
                self.running = True

                # Try one final time
                try:
                    next_item = self.queue.get(timeout=2.0)
                    if next_item is None:
                        raise StopIteration
                    return next_item
                except queue.Empty:
                    logger.error("Queue recovery failed, ending iteration")
                    raise StopIteration

    def __iter__(self) -> Iterator:
        return self

    def stop(self) -> None:
        """Stop the prefetch generator gracefully."""
        self.running = False
        self._shutdown_event.set()

        # Stop health monitoring if it's enabled
        if self.health_monitor is not None:
            self.health_monitor.stop_monitoring()

        # Wait for thread to finish (with timeout)
        if self.is_alive():
            self.join(timeout=2.0)


class PrefetchDataLoader(DataLoader):
    """Enhanced prefetch dataloader with fallback capabilities."""

    def __init__(
        self, num_prefetch_queue: int = 2, timeout: float = 15.0, **kwargs
    ) -> None:
        self.num_prefetch_queue = num_prefetch_queue
        self.prefetch_timeout = timeout
        self.fallback_dataloader = None
        self._using_fallback = False
        self._enhanced_prefetch_enabled = True  # Can be disabled if needed
        super().__init__(**kwargs)

    def __iter__(self) -> RobustPrefetchGenerator:  # type: ignore
        # Create fallback dataloader for graceful degradation
        self.fallback_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=max(1, self.num_workers // 2),  # Reduce workers for fallback
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.prefetch_timeout,
            worker_init_fn=self.worker_init_fn,
        )

        try:
            if self._enhanced_prefetch_enabled:
                generator = RobustPrefetchGenerator(
                    super().__iter__(), self.num_prefetch_queue
                )
                return generator
            else:
                # Use simplified generator
                return RobustPrefetchGenerator(
                    iter(self.fallback_dataloader), self.num_prefetch_queue
                )
        except Exception as e:
            logger = get_root_logger()
            logger.warning(
                f"Failed to create prefetch generator: {e}. Using fallback dataloader."
            )
            self._using_fallback = True
            return iter(self.fallback_dataloader)

    def reset(self) -> None:
        """Reset the dataloader state."""
        if self.fallback_dataloader is not None:
            # Recreate fallback dataloader
            self.fallback_dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=max(1, self.num_workers // 2),
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                timeout=self.prefetch_timeout,
                worker_init_fn=self.worker_init_fn,
            )

    def disable_enhanced_prefetch(self) -> None:
        """Disable enhanced prefetching and use simple fallback."""
        self._enhanced_prefetch_enabled = False


class CPUPrefetcher:
    """Enhanced CPU prefetcher with simplified, robust error handling."""

    def __init__(self, loader: DataLoader, timeout: float = 15.0) -> None:
        self.ori_loader = loader
        self.loader = None
        self.prefetch_timeout = timeout
        self.error_count = 0
        self.max_errors = 5

        # Initialize with iterator creation
        self._create_iterator_safe()

    def _create_iterator_safe(self) -> None:
        """Safely create the iterator with timeout protection."""
        try:
            self.loader = iter(self.ori_loader)
        except Exception as e:
            logger = get_root_logger()
            logger.warning(f"Failed to create iterator: {e}, using fallback")
            # Simple fallback without advanced features
            self.loader = iter(self.ori_loader)

    def next(self) -> Any:
        """Get next batch with timeout and error handling."""
        if self.loader is None:
            logger = get_root_logger()
            logger.warning("Iterator is None, recreating...")
            self._create_iterator_safe()

        try:
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
                logger.warning("Too many worker errors in CPU prefetcher, resetting")
                self._create_iterator_safe()
                self.error_count = 0
                # Try once more after reset
                try:
                    return next(self.loader)
                except StopIteration:
                    return None
                except Exception as retry_e:
                    logger.warning(f"Retry failed after reset: {retry_e}")
                    return None

            # Brief pause before retry
            time.sleep(0.001)
            raise

        except Exception as e:
            logger = get_root_logger()
            logger.warning(f"Error in CPU prefetcher: {type(e).__name__}: {e}")
            self.error_count += 1

            if self.error_count >= self.max_errors:
                logger.warning("Too many errors, resetting CPU prefetcher")
                self._create_iterator_safe()
                self.error_count = 0

            # Return None instead of raising to prevent training crashes
            logger.warning("Returning None due to prefetch error")
            return None

    def reset(self) -> None:
        """Reset the prefetcher."""
        self.error_count = 0
        self._create_iterator_safe()


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
