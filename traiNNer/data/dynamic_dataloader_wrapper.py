"""
Dynamic DataLoader and Dataset Wrappers for VRAM Management

These wrappers enable dynamic adjustment of batch_size and lq_size during training,
allowing the VRAM management system to make real-time adjustments that take effect
immediately without requiring a training restart.
"""

import ctypes
import logging
from collections.abc import Callable, Iterator
from multiprocessing import Value
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DynamicDatasetWrapper(Dataset):
    """
    Wrapper that enables dynamic updates to dataset parameters like gt_size/lq_size.

    This wrapper intercepts dataset access and applies current parameters dynamically,
    allowing the VRAM management system to adjust patch sizes during training.
    """

    def __init__(self, dataset: Dataset, initial_gt_size: int, scale: int = 2) -> None:
        self.dataset = dataset
        self.initial_gt_size = initial_gt_size
        self.scale = scale
        # Use shared memory for inter-process communication
        self._shared_gt_size = Value(ctypes.c_int, initial_gt_size)
        self._dataset_class_name = dataset.__class__.__name__

    def set_gt_size(self, new_gt_size: int) -> None:
        """Update gt_size dynamically."""
        old_gt_size = self.current_gt_size
        with self._shared_gt_size.get_lock():
            self._shared_gt_size.value = new_gt_size

        logger.info(
            f"Dynamic dataset: gt_size updated {old_gt_size} → {new_gt_size} "
            f"(lq_size: {new_gt_size // self.scale})"
        )

    @property
    def current_gt_size(self) -> int:
        with self._shared_gt_size.get_lock():
            return self._shared_gt_size.value

    @property
    def current_lq_size(self) -> int:
        return self.current_gt_size // self.scale

    def get_current_gt_size(self) -> int:
        """Get current gt_size."""
        return self.current_gt_size

    def get_current_lq_size(self) -> int:
        """Get current lq_size."""
        return self.current_lq_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        # Note: Dynamic logic for wrapper needs to be handled by wrapping the dataset's getitem
        # or relying on the dataset using shared values if we could inject them.
        # But DynamicDatasetWrapper wraps a generic dataset, so we can't easily inject.
        # This class is a fallback. The Mixins below are the primary method.
        return self.dataset[index]


class DynamicDataLoaderWrapper:
    """
    Wrapper that enables dynamic updates to DataLoader batch_size.

    This wrapper provides a transparent interface that allows batch_size adjustments
    during training without breaking the data loading pipeline.

    CRITICAL FIX: This wrapper now actually implements dynamic batch size control
    by implementing custom batch handling logic (slicing and accumulation).
    """

    def __init__(
        self,
        dataloader: DataLoader,
        initial_batch_size: int,
        update_callback: Callable[[int], None] | None = None,
        collate_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        self.dataloader = dataloader
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.update_callback = update_callback
        self.collate_fn = collate_fn

        # Store original batch size for reference
        self._original_batch_size = dataloader.batch_size

        # Buffer for pending batches to support slicing and accumulation
        self._pending_batches = []

        logger.info(
            f"Dynamic dataloader initialized with batch_size: {initial_batch_size} "
            f"(original: {self._original_batch_size})"
        )

    def set_batch_size(self, new_batch_size: int) -> None:
        """Update batch_size dynamically."""
        old_batch_size = self.current_batch_size
        self.current_batch_size = new_batch_size

        # We don't clear pending batches here - we want to preserve data
        # Just update the callback and log
        if self.update_callback:
            self.update_callback(new_batch_size)

        logger.info(
            f"Dynamic dataloader: batch_size ACTUALLY CHANGED {old_batch_size} → {new_batch_size} "
            f"(VRAM will now reflect this change)"
        )

    def get_current_batch_size(self) -> int:
        """Get current batch_size."""
        return self.current_batch_size

    def _get_batch_size(self, batch: Any) -> int:
        """Get the size of a batch (first dimension)."""
        if isinstance(batch, dict):
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    return value.size(0)
                if isinstance(value, (list, tuple)):
                    return len(value)
        elif isinstance(batch, torch.Tensor):
            return batch.size(0)
        elif isinstance(batch, (list, tuple)):
            # If we disable collation, the dataloader yields lists of samples
            # So a "batch" from the internal loader is a list
            # And its size is just len(batch)
            return len(batch)
        return 1

    def _get_sample_shape(self, sample: Any) -> tuple | None:
        """Get the shape of the main tensor in a sample to check compatibility."""
        if isinstance(sample, dict):
            # Usually 'lq' or 'gt' are the keys
            for k in ["lq", "img", "image"]:
                if k in sample and isinstance(sample[k], torch.Tensor):
                    return sample[k].shape
            # Fallback to any tensor
            for v in sample.values():
                if isinstance(v, torch.Tensor):
                    return v.shape
        elif isinstance(sample, torch.Tensor):
            return sample.shape
        elif isinstance(sample, (list, tuple)) and len(sample) > 0:
            if isinstance(sample[0], torch.Tensor):
                return sample[0].shape
        return None

    def _ensure_homogeneous_batch(self, items: list[Any]) -> list[list[Any]]:
        """
        Split a list of items into sub-lists where each sub-list has consistent shapes.
        This handles the case where crop size changes mid-stream.
        """
        if not items:
            return []

        batches = []
        current_batch = []
        current_shape = self._get_sample_shape(items[0])

        for item in items:
            shape = self._get_sample_shape(item)
            if (
                shape != current_shape
                and current_shape is not None
                and shape is not None
            ):
                # Shape changed! Seal current batch and start new one
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_shape = shape
            else:
                current_batch.append(item)

        if current_batch:
            batches.append(current_batch)

        return batches

    def _slice_batch(self, batch: Any, start: int, end: int) -> Any:
        """Slice a batch to get a subset."""
        if isinstance(batch, list):
            # Since we disabled collation, we work with lists
            return batch[start:end]

        # Fallback for legacy behavior/collated batches
        if isinstance(batch, dict):
            return {k: self._slice_batch(v, start, end) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch[start:end]
        elif isinstance(batch, (list, tuple)):
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                if hasattr(type(batch), "_make"):  # namedtuple
                    return type(batch)(
                        *(self._slice_batch(x, start, end) for x in batch)
                    )
                return type(batch)(self._slice_batch(x, start, end) for x in batch)
            return batch[start:end]
        return batch

    def _concat_batches(self, batches: list[Any]) -> Any:
        """Concatenate multiple batches."""
        if not batches:
            return None

        # If we work with lists (uncollated), just extend
        if isinstance(batches[0], list):
            result = []
            for b in batches:
                result.extend(b)
            return result

        # Fallback for tensor concatenation
        first = batches[0]
        if isinstance(first, dict):
            result = {}
            for k in first.keys():
                result[k] = self._concat_batches([b[k] for b in batches])
            return result
        elif isinstance(first, torch.Tensor):
            return torch.cat(batches, dim=0)
        return batches[0]

    def __len__(self) -> int:
        """Calculate length based on dynamic batch size."""
        original_length = len(self.dataloader)
        # Use initial_batch_size as fallback if original is None (e.g. when using batch_sampler)
        original_batch_size = self._original_batch_size or self.initial_batch_size
        current_batch_size = self.current_batch_size

        # Approximate total samples
        total_samples = original_length * original_batch_size

        # New length
        return max(1, int(total_samples / current_batch_size))

    def __iter__(self) -> Iterator[Any]:
        """Iterator that implements dynamic batch size control and late collation."""
        self._pending_batches = []
        iterator = iter(self.dataloader)
        exhausted = False

        while True:
            # Calculate current pending size
            pending_size = sum(self._get_batch_size(b) for b in self._pending_batches)

            # Fetch more data if needed
            while pending_size < self.current_batch_size and not exhausted:
                try:
                    batch = next(iterator)
                    # batch is now a list of samples (because we disabled collation)
                    self._pending_batches.append(batch)
                    pending_size += self._get_batch_size(batch)
                except StopIteration:
                    exhausted = True
                    break

            # Stop if no data left
            if pending_size == 0:
                break

            # If we don't have enough data and we are exhausted, yield what we have
            # Otherwise yield exactly current_batch_size
            target_size = min(self.current_batch_size, pending_size)

            # Extract batch of target_size (which is a LIST of samples)
            extracted_list = []
            collected_size = 0

            while collected_size < target_size:
                if not self._pending_batches:
                    break

                current = self._pending_batches[0]  # List of samples
                current_size = self._get_batch_size(current)
                needed = target_size - collected_size

                if current_size <= needed:
                    # Take the whole list
                    collected_size += current_size
                    extracted_list.extend(self._pending_batches.pop(0))
                else:
                    # Split the list
                    part1 = current[:needed]  # take needed items
                    part2 = current[needed:]  # remaining items

                    extracted_list.extend(part1)
                    self._pending_batches[0] = part2
                    collected_size += needed

            if extracted_list:
                # CRITICAL: Check for homogeneous shapes before collating
                # If sizes shifted (128 -> 136), we might have mixed items in extracted_list
                sub_batches = self._ensure_homogeneous_batch(extracted_list)

                for sub_batch in sub_batches:
                    if not sub_batch:
                        continue

                    # Now apply the collation function
                    if self.collate_fn:
                        try:
                            # Apply collation (stacking tensors, handling memory fmt)
                            yield self.collate_fn(sub_batch)
                        except RuntimeError as e:
                            logger.warning(
                                f"Collation failed for dynamic batch: {e}. Dropping batch."
                            )
                            continue
                    else:
                        # Return as list if no collator (fallback)
                        yield sub_batch
            else:
                break

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying dataloader."""
        return getattr(self.dataloader, name)


class PairedImageDatasetDynamicMixin:
    """
    Dynamic mixin for PairedImageDataset to support runtime gt_size updates.

    This mixin provides methods to dynamically update the cropping size used
    during training, enabling the VRAM management system to adjust patch sizes
    in real-time.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Use shared memory for inter-process communication
        initial_gt_size = self.opt.gt_size if self.opt.gt_size else 0
        self._shared_gt_size = Value(ctypes.c_int, initial_gt_size)

    def set_dynamic_gt_size(self, new_gt_size: int) -> None:
        """Update gt_size for dynamic cropping."""
        old_gt_size = self.current_gt_size
        with self._shared_gt_size.get_lock():
            self._shared_gt_size.value = new_gt_size

        logger.info(
            f"PairedImageDataset: Dynamic gt_size updated {old_gt_size} → {new_gt_size} "
            f"(lq_size: {new_gt_size // self.opt.scale})"
        )

    @property
    def current_gt_size(self) -> int:
        with self._shared_gt_size.get_lock():
            return self._shared_gt_size.value

    @property
    def current_lq_size(self) -> int:
        return self.current_gt_size // self.opt.scale

    def get_dynamic_gt_size(self) -> int:
        """Get current dynamic gt_size."""
        return self.current_gt_size

    def get_dynamic_lq_size(self) -> int:
        """Get current dynamic lq_size."""
        return self.current_lq_size

    def __getitem__(self, index: int):
        # Override the gt_size for cropping when in training phase
        # Use the shared value which is visible to all workers
        current_gt_size = self.current_gt_size

        if self.opt.phase == "train" and current_gt_size > 0:
            original_gt_size = getattr(self.opt, "gt_size", None)
            original_lq_size = getattr(self.opt, "lq_size", None)

            # Temporarily set the gt_size and lq_size for this getitem call
            self.opt.gt_size = current_gt_size
            self.opt.lq_size = current_gt_size // self.opt.scale

            try:
                result = super().__getitem__(index)
            finally:
                # Restore original gt_size and lq_size
                if original_gt_size is not None:
                    self.opt.gt_size = original_gt_size
                if original_lq_size is not None:
                    self.opt.lq_size = original_lq_size

            return result
        else:
            return super().__getitem__(index)


class PairedVideoDatasetDynamicMixin:
    """
    Dynamic mixin for PairedVideoDataset to support runtime gt_size updates.

    Similar to PairedImageDatasetDynamicMixin but for video datasets.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Use shared memory for inter-process communication
        initial_gt_size = self.gt_size if self.gt_size else 0
        self._shared_gt_size = Value(ctypes.c_int, initial_gt_size)

    def set_dynamic_gt_size(self, new_gt_size: int) -> None:
        """Update gt_size for dynamic cropping."""
        old_gt_size = self.current_gt_size
        with self._shared_gt_size.get_lock():
            self._shared_gt_size.value = new_gt_size

        logger.info(
            f"PairedVideoDataset: Dynamic gt_size updated {old_gt_size} → {new_gt_size} "
            f"(lq_size: {new_gt_size // self.opt.scale})"
        )

    @property
    def current_gt_size(self) -> int:
        with self._shared_gt_size.get_lock():
            return self._shared_gt_size.value

    @property
    def current_lq_size(self) -> int:
        return self.current_gt_size // self.opt.scale

    def get_dynamic_gt_size(self) -> int:
        """Get current dynamic gt_size."""
        return self.current_gt_size

    def get_dynamic_lq_size(self) -> int:
        """Get current dynamic lq_size."""
        return self.current_lq_size


def create_dynamic_dataset(
    dataset: Dataset, gt_size: int, scale: int = 2
) -> DynamicDatasetWrapper:
    """Create a dynamic wrapper around a dataset."""
    return DynamicDatasetWrapper(dataset, gt_size, scale)


def create_dynamic_dataloader(
    dataloader: DataLoader,
    batch_size: int,
    update_callback: Callable[[int], None] | None = None,
    collate_fn: Callable[[list[Any]], Any] | None = None,
) -> DynamicDataLoaderWrapper:
    """Create a dynamic wrapper around a dataloader."""
    return DynamicDataLoaderWrapper(dataloader, batch_size, update_callback, collate_fn)


def patch_dataset_for_dynamic_updates(dataset: Dataset) -> Dataset:
    """
    Patch a dataset class to support dynamic updates.

    This function dynamically modifies dataset classes to support runtime
    parameter updates for VRAM management.
    """
    from traiNNer.data.paired_image_dataset import PairedImageDataset
    from traiNNer.data.paired_video_dataset import PairedVideoDataset

    if isinstance(dataset, PairedImageDataset):
        # Create a dynamic version of PairedImageDataset
        class PairedImageDatasetDynamic(
            PairedImageDatasetDynamicMixin, PairedImageDataset
        ):
            pass

        # Create instance of the dynamic class
        dynamic_dataset = PairedImageDatasetDynamic(dataset.opt)
        # Copy all attributes from original dataset
        for attr_name in dir(dataset):
            if not attr_name.startswith("_") and hasattr(dataset, attr_name):
                try:
                    setattr(dynamic_dataset, attr_name, getattr(dataset, attr_name))
                except (AttributeError, TypeError):
                    pass
        return dynamic_dataset

    elif isinstance(dataset, PairedVideoDataset):
        # Create a dynamic version of PairedVideoDataset
        class PairedVideoDatasetDynamic(
            PairedVideoDatasetDynamicMixin, PairedVideoDataset
        ):
            pass

        # Create instance of the dynamic class
        dynamic_dataset = PairedVideoDatasetDynamic(dataset.opt)
        # Copy all attributes from original dataset
        for attr_name in dir(dataset):
            if not attr_name.startswith("_") and hasattr(dataset, attr_name):
                try:
                    setattr(dynamic_dataset, attr_name, getattr(dataset, attr_name))
                except (AttributeError, TypeError):
                    pass
        return dynamic_dataset

    else:
        # For other dataset types, wrap with DynamicDatasetWrapper
        logger.warning(
            f"Dataset type {dataset.__class__.__name__} not natively supported for dynamic updates. "
            f"Using basic wrapper."
        )
        return create_dynamic_dataset(
            dataset,
            gt_size=getattr(dataset.opt, "gt_size", 128),
            scale=getattr(dataset.opt, "scale", 2),
        )
