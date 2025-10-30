#!/usr/bin/env python3
"""
ParagonSR Model Fusion Utility (Safetensors Edition)
Author: Philip Hofmann

Description:
This script loads a trained ParagonSR checkpoint (in .safetensors format),
applies the permanent fusion logic, and saves the new, simplified state_dict
back into the .safetensors format for consistency.

Usage:
python -m scripts.paragonsr.fuse_model --input path/to/model.safetensors --output path/to/fused_model.safetensors --scale 4
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# --- Import both load_file and save_file ---
from safetensors.torch import load_file, save_file

# This import must be absolute to work when run as a module
from traiNNer.archs.paragonsr_arch import paragonsr_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse ParagonSR model for deployment")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .safetensors checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the fused model will be saved",
    )
    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        help="Scale factor for the model (e.g., 2, 4, 8)",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    if not args.input.endswith(".safetensors"):
        print("Error: Input file must be a .safetensors file.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate scale factor
    if args.scale not in [1, 2, 3, 4, 6, 8, 16]:
        print(
            f"Error: Invalid scale factor '{args.scale}'. Must be one of: 1, 2, 3, 4, 6, 8, 16"
        )
        sys.exit(1)

    # Determine model variant based on filename
    model_func = paragonsr_s  # Default to small variant

    # Try to infer model size from filename
    input_filename = Path(args.input).stem.lower()
    if "tiny" in input_filename:
        from traiNNer.archs.paragonsr_arch import paragonsr_tiny

        model_func = paragonsr_tiny
    elif "xs" in input_filename:
        from traiNNer.archs.paragonsr_arch import paragonsr_xs

        model_func = paragonsr_xs
    elif "m" in input_filename and "medium" not in input_filename:
        from traiNNer.archs.paragonsr_arch import paragonsr_m

        model_func = paragonsr_m
    elif "l" in input_filename and "large" not in input_filename:
        from traiNNer.archs.paragonsr_arch import paragonsr_l

        model_func = paragonsr_l
    elif "xl" in input_filename:
        from traiNNer.archs.paragonsr_arch import paragonsr_xl

        model_func = paragonsr_xl

    print("--- Starting ParagonSR Model Fusion ---")

    # 1. Initialize the model structure
    model = model_func(scale=args.scale)
    print(
        f"Initialized '{model.__class__.__name__}' architecture with scale factor {args.scale}."
    )

    # 2. Load the trained weights from the .safetensors file
    print(f"Loading training weights from: {args.input}")
    try:
        state_dict = load_file(args.input)
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 3. Switch the model to evaluation mode
    model.eval()
    print("Model switched to evaluation mode.")

    # 4. Call the permanent fusion method
    model.fuse_for_release()
    print("Fusion complete. The model architecture has been permanently simplified.")

    # 5. Save the new, fused state_dict as a .safetensors file
    # This is a simple replacement for torch.save()
    try:
        save_file(model.state_dict(), args.output)
        print(f"\nSuccessfully saved fused model to: {args.output}")
    except Exception as e:
        print(f"Error saving fused model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
