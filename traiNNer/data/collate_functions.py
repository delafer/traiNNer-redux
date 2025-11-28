"""
Custom collate functions for handling memory format compatibility.

This module provides collate functions that handle memory format conversion
before data reaches DataLoader's pin_memory operation, resolving bf16 +
channels_last + pin_memory compatibility issues.
"""

from typing import Any, Dict, List, Union

import torch
from torch.utils.data import default_collate

from traiNNer.utils.redux_options import ReduxOptions


def channels_last_collate_fn(
    batch: list[dict[str, Any]], opt: ReduxOptions
) -> dict[str, torch.Tensor]:
    """
    Custom collate function that converts tensors to channels_last memory format
    before DataLoader operations, preventing pin_memory compatibility issues.

    Args:
        batch: List of data samples from dataset
        opt: ReduxOptions containing AMP and memory format settings

    Returns:
        Dict with tensors in appropriate memory format
    """
    if not opt.use_amp or not opt.use_channels_last:
        # Use default collate if no memory format conversion needed
        return default_collate(batch)

    # Use channels_last memory format for AMP training
    memory_format = torch.channels_last
    collated = default_collate(batch)

    # Convert tensors to channels_last memory format
    for key, value in collated.items():
        if (
            torch.is_tensor(value) and value.dim() >= 3
        ):  # Only convert tensors with 3+ dims
            if not value.is_contiguous(memory_format=memory_format):
                collated[key] = value.contiguous(memory_format=memory_format)

    return collated


def create_collate_function(opt: ReduxOptions):
    """
    Create appropriate collate function based on training settings.

    Args:
        opt: ReduxOptions containing training configuration

    Returns:
        Collate function to use with DataLoader
    """
    if opt.use_amp and opt.use_channels_last:
        # Use custom collate function for channels_last compatibility
        return lambda batch: channels_last_collate_fn(batch, opt)
    else:
        # Use default collate function
        return default_collate
