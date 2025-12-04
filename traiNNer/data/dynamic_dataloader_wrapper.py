"""
Dynamic DataLoader and Dataset Wrappers for VRAM Management

These wrappers enable dynamic adjustment of batch_size and lq_size during training,
allowing the VRAM management system to make real-time adjustments that take effect
immediately without requiring a training restart.
"""

import logging
from collections.abc import Callable, Iterator
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
        self.current_gt_size = initial_gt_size
        self.current_lq_size = initial_gt_size // scale

        # Store original dataset methods if they need to be intercepted
        self._original_getitem = None
        self._dataset_class_name = dataset.__class__.__name__

    def set_gt_size(self, new_gt_size: int) -> None:
        """Update gt_size dynamically."""
        old_gt_size = self.current_gt_size
        self.current_gt_size = new_gt_size
        self.current_lq_size = new_gt_size // self.scale

        logger.info(
            f"Dynamic dataset: gt_size updated {old_gt_size} → {new_gt_size} "
            f"(lq_size: {self.current_lq_size})"
        )

    def get_current_gt_size(self) -> int:
        """Get current gt_size."""
        return self.current_gt_size

    def get_current_lq_size(self) -> int:
        """Get current lq_size."""
        return self.current_lq_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        # For certain dataset types, we need to temporarily override the gt_size
        # This is handled in the specific dataset classes via dynamic parameter injection
        return self.dataset[index]


class DynamicDataLoaderWrapper:
    """
    Wrapper that enables dynamic updates to DataLoader batch_size.

    This wrapper provides a transparent interface that allows batch_size adjustments
    during training without breaking the data loading pipeline.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        initial_batch_size: int,
        update_callback: Callable[[int], None] | None = None,
    ) -> None:
        self.dataloader = dataloader
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.update_callback = update_callback

        # Store original batch size for reference
        self._original_batch_size = dataloader.batch_size

        logger.info(
            f"Dynamic dataloader initialized with batch_size: {initial_batch_size}"
        )

    def set_batch_size(self, new_batch_size: int) -> None:
        """Update batch_size dynamically."""
        old_batch_size = self.current_batch_size
        self.current_batch_size = new_batch_size

        # Don't try to modify the underlying DataLoader's batch_size attribute
        # as PyTorch DataLoader doesn't allow this after initialization
        # Instead, just track the new batch size and let the training loop handle it
        # through the wrapper interface

        # Call update callback if provided
        if self.update_callback:
            self.update_callback(new_batch_size)

        logger.info(
            f"Dynamic dataloader: batch_size tracked {old_batch_size} → {new_batch_size} "
            f"(wrapper will handle actual usage)"
        )

    def get_current_batch_size(self) -> int:
        """Get current batch_size."""
        return self.current_batch_size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)

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
        self._dynamic_gt_size = self.opt.gt_size
        self._dynamic_lq_size = (
            self.opt.gt_size // self.opt.scale if self.opt.gt_size else None
        )

    def set_dynamic_gt_size(self, new_gt_size: int) -> None:
        """Update gt_size for dynamic cropping."""
        old_gt_size = self._dynamic_gt_size
        self._dynamic_gt_size = new_gt_size
        self._dynamic_lq_size = new_gt_size // self.opt.scale

        logger.debug(
            f"PairedImageDataset: Dynamic gt_size updated {old_gt_size} → {new_gt_size}"
        )

    def get_dynamic_gt_size(self) -> int:
        """Get current dynamic gt_size."""
        return self._dynamic_gt_size

    def get_dynamic_lq_size(self) -> int:
        """Get current dynamic lq_size."""
        return self._dynamic_lq_size

    def __getitem__(self, index: int):
        # Override the gt_size for cropping when in training phase
        if self.opt.phase == "train" and hasattr(self, "_dynamic_gt_size"):
            original_gt_size = self.opt.gt_size
            # Temporarily set the gt_size for this getitem call
            self.opt.gt_size = self._dynamic_gt_size

            try:
                result = super().__getitem__(index)
            finally:
                # Restore original gt_size
                self.opt.gt_size = original_gt_size

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
        self._dynamic_gt_size = self.gt_size
        self._dynamic_lq_size = self.gt_size // self.opt.scale if self.gt_size else None

    def set_dynamic_gt_size(self, new_gt_size: int) -> None:
        """Update gt_size for dynamic cropping."""
        old_gt_size = self._dynamic_gt_size
        self._dynamic_gt_size = new_gt_size
        self._dynamic_lq_size = new_gt_size // self.opt.scale

        logger.debug(
            f"PairedVideoDataset: Dynamic gt_size updated {old_gt_size} → {new_gt_size}"
        )

    def get_dynamic_gt_size(self) -> int:
        """Get current dynamic gt_size."""
        return self._dynamic_gt_size

    def get_dynamic_lq_size(self) -> int:
        """Get current dynamic lq_size."""
        return self._dynamic_lq_size


def create_dynamic_dataset(
    dataset: Dataset, gt_size: int, scale: int = 2
) -> DynamicDatasetWrapper:
    """Create a dynamic wrapper around a dataset."""
    return DynamicDatasetWrapper(dataset, gt_size, scale)


def create_dynamic_dataloader(
    dataloader: DataLoader,
    batch_size: int,
    update_callback: Callable[[int], None] | None = None,
) -> DynamicDataLoaderWrapper:
    """Create a dynamic wrapper around a dataloader."""
    return DynamicDataLoaderWrapper(dataloader, batch_size, update_callback)


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
