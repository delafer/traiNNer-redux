"""
Grid Sample Compatibility Patch for PyTorch
Fixes the mode parameter compatibility issue between newer PyTorch versions and libraries like spandrel
"""

import torch
import torch.nn.functional as F
from torch.nn import functional

# Store the original function
_original_grid_sample = F.grid_sample


def _patched_grid_sample(
    input, grid, mode=None, padding_mode=None, align_corners=None, **kwargs
):
    """
    Patched version of torch.nn.functional.grid_sample that handles the 'mode' parameter
    gracefully for compatibility with libraries that use deprecated parameter names.
    """
    # Create a clean arguments dict for the actual function call
    clean_kwargs = {}

    # Handle the deprecated mode parameter
    if mode is not None:
        if mode != "bilinear":
            import warnings

            warnings.warn(
                f"grid_sample: Only 'bilinear' mode is supported, ignoring mode='{mode}'",
                stacklevel=2,
            )
        # mode is ignored as it's deprecated (bilinear is default in newer PyTorch)

    # Handle supported parameters
    if padding_mode is not None:
        clean_kwargs["padding_mode"] = padding_mode
    if align_corners is not None:
        clean_kwargs["align_corners"] = align_corners

    # Add any additional kwargs
    for k, v in kwargs.items():
        clean_kwargs[k] = v

    # Call the original function with clean parameters
    return _original_grid_sample(input, grid, **clean_kwargs)


# Replace the function globally
torch.nn.functional.grid_sample = _patched_grid_sample
F.grid_sample = _patched_grid_sample


def enable_grid_sample_fix() -> None:
    """Enable the grid_sample compatibility fix"""
    global _original_grid_sample
    # Already applied above


def disable_grid_sample_fix() -> None:
    """Disable the grid_sample compatibility fix"""
    global _original_grid_sample
    torch.nn.functional.grid_sample = _original_grid_sample
    F.grid_sample = _original_grid_sample


# Apply the fix immediately
enable_grid_sample_fix()
