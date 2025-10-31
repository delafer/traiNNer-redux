#!/usr/bin/env python3
"""
ParagonSR Model Comparison Tool
Author: Philip Hofmann

Description:
This script compares the visual output of your trained ParagonSR checkpoint
against the ONNX exported models to identify if checkerboard/pixelation issues
come from training or ONNX conversion.

Usage:
python -m scripts.paragonsr.compare_models --checkpoint path/to/trained_model.safetensors --onnx path/to/model_fp32.onnx --images path/to/validation_images/ --scale 4 --model_variant s --input_size 256

This will generate comparison images showing:
- Original LR input
- PyTorch inference (trained checkpoint)
- ONNX inference
- Pixel difference map
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import traiNNer modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from traiNNer.archs.paragonsr_arch import paragonsr_s


def get_model_variant(model_name: str):
    """Get the ParagonSR model variant function from name."""
    model_name = model_name.lower()

    variants = {
        "tiny": ("traiNNer.archs.paragonsr_arch", "paragonsr_tiny"),
        "xs": ("traiNNer.archs.paragonsr_arch", "paragonsr_xs"),
        "s": ("traiNNer.archs.paragonsr_arch", "paragonsr_s"),
        "m": ("traiNNer.archs.paragonsr_arch", "paragonsr_m"),
        "l": ("traiNNer.archs.paragonsr_arch", "paragonsr_l"),
        "xl": ("traiNNer.archs.paragonsr_arch", "paragonsr_xl"),
    }

    if model_name not in variants:
        raise ValueError(
            f"Unknown model variant '{model_name}'. Choose from: {list(variants.keys())}"
        )

    module_name, func_name = variants[model_name]
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def load_torch_model(checkpoint_path: str, model_func, scale: int):
    """Load and prepare ParagonSR model for inference."""
    model = model_func(scale=scale)
    model.eval()

    # Load checkpoint
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)

    return model


def load_onnx_model(onnx_path: str):
    """Load ONNX model for inference."""
    session = ort.InferenceSession(onnx_path)
    return session


def preprocess_image(image_path: str, input_size: int = 64):
    """Load and preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize to input size (for consistent testing)
    image = image.resize((input_size, input_size), Image.BICUBIC)

    # Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor, image


def run_pytorch_inference(model, input_tensor):
    """Run inference with PyTorch model."""
    with torch.no_grad():
        output = model(input_tensor)
        # Denormalize (fix broadcasting issue)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(output.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(output.device)
        output = output * std + mean
        output = torch.clamp(output, 0, 1)
        return output


def run_onnx_inference(session, input_tensor):
    """Run inference with ONNX model."""
    # Convert tensor to numpy (keep NCHW format as exported)
    input_np = input_tensor.cpu().numpy()  # Shape (1, 3, 64, 64)

    # Denormalize (fix broadcasting for both input and output)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    input_np = input_np * std + mean
    input_np = np.clip(input_np, 0, 1)

    # Keep NCHW format (don't transpose to NHWC)
    # input_np shape remains (1, 3, 64, 64)

    # CRITICAL: Convert to float32 for ONNX compatibility
    input_np = input_np.astype(np.float32)

    # Run inference
    outputs = session.run(None, {"input": input_np})

    # Process output (NCHW format expected)
    output = outputs[0]  # First output, shape (1, 3, 256, 256) for 4x
    output = np.clip(output, 0, 1)

    # Convert back to tensor format (NCHW)
    output_tensor = torch.from_numpy(output).float()
    return output_tensor


def save_comparison_image(
    lr_img, pytorch_output, onnx_output, diff_map, output_path
) -> None:
    """Save side-by-side comparison image."""

    # Convert images to PIL for easy saving
    def tensor_to_pil(tensor):
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        transform = transforms.ToPILImage()
        return transform(tensor)

    # Handle LR image (could be PIL Image or tensor)
    if isinstance(lr_img, Image.Image):
        lr_pil = lr_img
    else:
        lr_pil = tensor_to_pil(lr_img)

    # Convert tensors to PIL
    pytorch_pil = tensor_to_pil(pytorch_output)
    onnx_pil = tensor_to_pil(onnx_output)
    diff_pil = tensor_to_pil(diff_map)

    # Create comparison grid
    images = [lr_pil, pytorch_pil, onnx_pil, diff_pil]

    # Save as 2x2 grid
    grid_img = Image.new("RGB", (images[0].width * 2, images[0].height * 2))
    grid_img.paste(images[0], (0, 0))  # LR
    grid_img.paste(images[1], (images[0].width, 0))  # PyTorch
    grid_img.paste(images[2], (0, images[0].height))  # ONNX
    grid_img.paste(images[3], (images[0].width, images[0].height))  # Diff

    grid_img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ParagonSR trained checkpoint vs ONNX model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained .safetensors checkpoint",
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="Path to ONNX model (.onnx file)"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Output directory for comparison images",
    )
    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 6, 8, 16],
        help="Model scale factor",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        required=True,
        choices=["tiny", "xs", "s", "m", "l", "xl"],
        help="ParagonSR model variant",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
        help="Input image size for testing (default: 64)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.onnx):
        print(f"Error: ONNX file '{args.onnx}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.images):
        print(f"Error: Images directory '{args.images}' does not exist.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🚀 ParagonSR Model Comparison Tool")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX Model: {args.onnx}")
    print(f"Images: {args.images}")
    print(f"Output: {args.output_dir}")
    print("=" * 50)

    # Load models
    print("Loading models...")
    try:
        model_func = get_model_variant(args.model_variant)
        torch_model = load_torch_model(args.checkpoint, model_func, args.scale)
        onnx_session = load_onnx_model(args.onnx)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

    # Process images
    image_files = [
        f
        for f in os.listdir(args.images)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"❌ No image files found in '{args.images}'")
        sys.exit(1)

    print(f"Processing {len(image_files)} images...")

    for i, image_file in enumerate(image_files):
        try:
            image_path = os.path.join(args.images, image_file)
            print(f"  Processing {image_file} ({i + 1}/{len(image_files)})")

            # Load and preprocess
            input_tensor, original_lr = preprocess_image(image_path, args.input_size)

            # Run PyTorch inference
            pytorch_output = run_pytorch_inference(torch_model, input_tensor)

            # Run ONNX inference
            onnx_output = run_onnx_inference(onnx_session, input_tensor)

            # Calculate pixel difference
            diff_map = torch.abs(pytorch_output - onnx_output)

            # Scale up the output for visualization (since we're testing with downscaled images)
            if args.scale > 1:
                # This is just for visualization - in real testing you'd use proper SR images
                pytorch_output = torch.nn.functional.interpolate(
                    pytorch_output,
                    scale_factor=args.scale,
                    mode="bilinear",
                    align_corners=False,
                )
                onnx_output = torch.nn.functional.interpolate(
                    onnx_output,
                    scale_factor=args.scale,
                    mode="bilinear",
                    align_corners=False,
                )
                diff_map = torch.nn.functional.interpolate(
                    diff_map,
                    scale_factor=args.scale,
                    mode="bilinear",
                    align_corners=False,
                )

            # Save comparison
            base_name = Path(image_file).stem
            comparison_path = output_dir / f"{base_name}_comparison.png"
            save_comparison_image(
                original_lr, pytorch_output, onnx_output, diff_map, comparison_path
            )

            print(f"    ✅ Saved: {comparison_path}")

        except Exception as e:
            print(f"    ❌ Error processing {image_file}: {e}")
            continue

    print("\n🎉 Comparison complete!")
    print(f"Results saved in: {args.output_dir}")
    print("\nVisual inspection tips:")
    print("- Look for checkerboard patterns in both PyTorch and ONNX outputs")
    print(
        "- Check if the difference map shows random noise (good) or structured artifacts (bad)"
    )
    print("- If PyTorch has issues, the problem is in training")
    print("- If only ONNX has issues, the problem is in conversion")


if __name__ == "__main__":
    main()
