#!/usr/bin/env python3
"""
Simple ParagonSR Test Script
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from traiNNer.archs.paragonsr_arch import paragonsr_s


def test_single_image() -> None:
    """Test with a single image to debug issues."""
    print("=== ParagonSR Single Image Test ===")

    # Test paths
    checkpoint = "experiments/4x_ParagonSR_S/models/4xParagonSR_S_pretrain.safetensors"
    onnx_model = "experiments/4x_ParagonSR_S/models/release/4xParagonSR_S_pretrain_fused_op18_fp32.onnx"
    test_image = "/home/phips/Documents/dataset/cc0/val_x4/1.png"

    print(f"Checkpoint: {checkpoint}")
    print(f"ONNX: {onnx_model}")
    print(f"Test Image: {test_image}")

    # Check if files exist
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found: {checkpoint}")
        return
    if not os.path.exists(onnx_model):
        print(f"❌ ONNX model not found: {onnx_model}")
        return
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return

    print("✅ All files found")

    # Test 1: Load PyTorch model
    print("\n--- Test 1: Loading PyTorch model ---")
    try:
        model = paragonsr_s(scale=4)
        model.eval()
        state_dict = load_file(checkpoint)
        model.load_state_dict(state_dict)
        print("✅ PyTorch model loaded successfully")
    except Exception as e:
        print(f"❌ PyTorch model failed: {e}")
        return

    # Test 2: Load ONNX model
    print("\n--- Test 2: Loading ONNX model ---")
    try:
        session = ort.InferenceSession(onnx_model)
        print("✅ ONNX model loaded successfully")
    except Exception as e:
        print(f"❌ ONNX model failed: {e}")
        return

    # Test 3: Process image
    print("\n--- Test 3: Processing image ---")
    try:
        # Load image
        image = Image.open(test_image).convert("RGB")
        print(f"Original image size: {image.size}")

        # Resize for testing
        image = image.resize((64, 64), Image.BICUBIC)
        print(f"Resized image size: {image.size}")

        # Convert to tensor and normalize
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        print(f"Input tensor shape: {input_tensor.shape}")

        # Test PyTorch inference
        print("\n--- Test 4: PyTorch inference ---")
        with torch.no_grad():
            output = model(input_tensor)
            print(f"PyTorch output shape: {output.shape}")

        # Test ONNX inference
        print("\n--- Test 5: ONNX inference ---")
        # Convert to numpy and denormalize
        input_np = input_tensor.cpu().numpy()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).numpy()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).numpy()
        input_np = input_np * std + mean
        input_np = np.clip(input_np, 0, 1)
        input_np = input_np.astype(np.float32)

        print(f"ONNX input shape: {input_np.shape}")

        outputs = session.run(None, {"input": input_np})
        onnx_output = outputs[0]
        print(f"ONNX output shape: {onnx_output.shape}")

        # Test saving
        print("\n--- Test 6: Saving comparison ---")
        output_dir = Path("debug_results")
        output_dir.mkdir(exist_ok=True)

        # Convert tensors to PIL for comparison
        def tensor_to_pil(tensor):
            tensor = tensor.squeeze(0).cpu()
            tensor = torch.clamp(tensor, 0, 1)
            return transforms.ToPILImage()(tensor)

        # Save individual images
        transforms.ToPILImage()(image).save(output_dir / "input.png")
        tensor_to_pil(output).save(output_dir / "pytorch_output.png")

        onnx_tensor = torch.from_numpy(onnx_output).float()
        tensor_to_pil(onnx_tensor).save(output_dir / "onnx_output.png")

        # Create comparison image
        comparison = Image.new("RGB", (256, 512))  # 2x2 grid: 2*128 x 2*256
        comparison.paste(transforms.ToPILImage()(image), (0, 0))
        comparison.paste(tensor_to_pil(output), (128, 0))
        comparison.paste(tensor_to_pil(onnx_tensor), (0, 256))

        diff_tensor = torch.abs(output - onnx_tensor)
        comparison.paste(tensor_to_pil(diff_tensor), (128, 256))

        comparison.save(output_dir / "comparison.png")
        print(f"✅ Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_single_image()
