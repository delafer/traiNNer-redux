#!/usr/bin/env python3
"""
ParagonSR2 Release ONNX Converter
=================================

Creates production-ready, dynamic ONNX models (FP32, FP16, INT8) for ParagonSR2.
Features:
- Automatic model fusion (ReparamConvV2)
- Dynamic input shapes (Batch, Height, Width)
- TensorRT and ONNX Runtime compatibility
- Intelligent FP16 conversion (automatic mixed precision)
- High-quality INT8 quantization (QDQ format) with calibration
- Comprehensive validation against PyTorch baseline

Usage:
    python scripts/paragonsr2/convert_onnx_release.py \
        --checkpoint models/my_model.safetensors \
        --arch paragonsr2_static_s \
        --scale 2 \
        --output release_output \
        --calib_dir datasets/calibration \
        --val_dir datasets/validation

Author: Kilo Code (traiNNer-redux)
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import checker
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image

# Import architecture to ensure registry is populated
try:
    # Try importing from local project structure
    sys.path.append(str(Path(__file__).parents[2]))
    import traiNNer.archs.paragonsr2_static_arch
    from traiNNer.utils.registry import ARCH_REGISTRY
except ImportError:
    print(
        "Error: Could not import traiNNer modules. Make sure you are in the project root."
    )
    sys.exit(1)


# =============================================================================
# UTILITIES
# =============================================================================


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    if border > 0:
        img1 = img1[border:-border, border:-border, :]
        img2 = img2[border:-border, border:-border, :]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def preprocess_image(
    image_path: Path, input_size: tuple[int, int] | None, norm_type: str = "01"
) -> np.ndarray:
    """
    Preprocess image for model input.

    Args:
        image_path: Path to image file
        input_size: (height, width) to resize to
        norm_type: '01' for [0, 1] range, 'm11' for [-1, 1] range
    """
    img = Image.open(image_path).convert("RGB")

    # Resize if input_size is provided (for calibration/fixed testing)
    if input_size is not None:
        img = img.resize((input_size[1], input_size[0]), Image.Resampling.BICUBIC)

    # Convert to numpy float32 CHW
    img_array = np.array(img).astype(np.float32) / 255.0

    if norm_type == "m11":
        img_array = (img_array - 0.5) / 0.5

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


def postprocess_output(output: np.ndarray, norm_type: str = "01") -> np.ndarray:
    """Convert model output back to uint8 HWC image."""
    output = output.squeeze(0)  # Remove batch
    output = np.transpose(output, (1, 2, 0))  # CHW -> HWC

    if norm_type == "m11":
        output = (output * 0.5) + 0.5

    output = np.clip(output, 0, 1)
    output = (output * 255.0).round().astype(np.uint8)
    return output


class ParagonCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for INT8 quantization."""

    def __init__(
        self,
        calib_dir: str,
        input_size: tuple[int, int],
        input_name: str,
        norm_type: str = "01",
        limit: int | None = None,
    ) -> None:
        self.image_paths = sorted(Path(calib_dir).glob("*"))
        # Filter for images
        self.image_paths = [
            p
            for p in self.image_paths
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        ]

        if not self.image_paths:
            raise ValueError(f"No images found in {calib_dir}")

        if limit is not None:
            self.image_paths = self.image_paths[:limit]
            print(f"      Using {len(self.image_paths)} images for calibration")

        self.input_size = input_size
        self.input_name = input_name
        self.norm_type = norm_type
        self.enum_data = iter(self.image_paths)

    def get_next(self) -> dict[str, Any] | None:
        try:
            image_path = next(self.enum_data)
            input_data = preprocess_image(image_path, self.input_size, self.norm_type)
            return {self.input_name: input_data}
        except StopIteration:
            return None


# =============================================================================
# CONVERTER CLASS
# =============================================================================


class ParagonConverter:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = {
            "model": args.arch,
            "scale": args.scale,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": {},
        }

    def load_model(self) -> torch.nn.Module:
        """Load and fuse the PyTorch model."""
        print(f"\n[1/5] Loading model: {self.args.arch}")

        arch_fn = ARCH_REGISTRY.get(self.args.arch)
        if not arch_fn:
            raise ValueError(f"Architecture {self.args.arch} not found in registry.")

        model = arch_fn(scale=self.args.scale)

        # Load weights
        print(f"      Loading checkpoint: {self.args.checkpoint}")
        if self.args.checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(self.args.checkpoint)
        else:
            state_dict = torch.load(self.args.checkpoint, map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        # Fuse layers
        if hasattr(model, "fuse_for_release"):
            print("      Fusing ReparamConvV2 blocks for inference...")
            model.fuse_for_release()

        return model

    def export_fp32(self, model: torch.nn.Module) -> Path:
        """Export standard FP32 ONNX model."""
        print("\n[2/5] Exporting FP32 ONNX...")
        output_path = self.output_dir / f"{self.args.arch}_fp32.onnx"

        # Dynamic axes for variable input size
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

        # Dummy input (fixed size for export tracing)
        dummy_input = torch.randn(1, 3, 64, 64, device=self.device)

        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
        )

        # Optimize graph
        onnx_model = onnx.load(str(output_path))
        checker.check_model(onnx_model)
        onnx.save(onnx_model, str(output_path))

        print(f"      Saved to: {output_path}")
        self.report["results"]["fp32"] = {
            "path": str(output_path),
            "size_mb": output_path.stat().st_size / (1024 * 1024),
        }
        return output_path

    def convert_fp16(self, fp32_path: Path) -> Path:
        """Convert to FP16 with automatic mixed precision."""
        print("\n[3/5] Converting to FP16...")
        output_path = self.output_dir / f"{self.args.arch}_fp16.onnx"

        model = onnx.load(str(fp32_path))

        # Convert to FP16, keeping IO as float32 for compatibility
        # We use a comprehensive block list to prevent converting ops that are sensitive to precision
        # or involved in control flow (Loops) which often cause type mismatches in ONNX Runtime.

        print("      Applying FP16 conversion with protected operators...")

        # Extended block list for maximum compatibility
        # These operators will remain in FP32
        block_list = [
            "Resize",
            "Upsample",
            "GridSample",
            "Loop",
            "SequenceInsert",
            "SequenceAt",
            "SequenceEmpty",
            "SequenceErase",
            "SequenceLength",
            "SequenceConstruct",
            "Range",
            "Tile",
            "Expand",
            "Softmax",  # Often better in FP32 for stability
            "LayerNormalization",  # Often better in FP32
        ]

        try:
            fp16_model = float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=block_list,
            )
        except Exception as e:
            print(
                f"      Warning: Primary FP16 conversion failed ({e}). Retrying with safe mode..."
            )
            # Fallback: disable shape inference which sometimes causes issues
            fp16_model = float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                disable_shape_infer=True,
                op_block_list=block_list,
            )

        onnx.save(fp16_model, str(output_path))
        print(f"      Saved to: {output_path}")
        self.report["results"]["fp16"] = {
            "path": str(output_path),
            "size_mb": output_path.stat().st_size / (1024 * 1024),
        }
        return output_path

    def convert_int8(self, fp32_path: Path) -> Path | None:
        """Convert to INT8 using QDQ format and calibration."""
        if not self.args.calib_dir:
            print(
                "\n[4/5] Skipping INT8 conversion (no calibration directory provided)"
            )
            return None

        print("\n[4/5] Converting to INT8 (QDQ)...")
        output_path = self.output_dir / f"{self.args.arch}_int8.onnx"

        # Calibration reader
        dr = ParagonCalibrationDataReader(
            self.args.calib_dir,
            (self.args.calib_size, self.args.calib_size),
            "input",
            self.args.norm,
            limit=self.args.calib_count,
        )

        # Quantize
        # QuantFormat.QDQ is best for TensorRT and modern ONNX Runtime
        # PerChannel=True usually gives better accuracy for CNNs
        # Disable per-channel quantization for weights to avoid "Axis out of range" errors
        quantize_static(
            str(fp32_path),
            str(output_path),
            dr,
            quant_format=QuantFormat.QDQ,
            per_channel=False,  # Fixed: Disable per-channel to avoid axis errors
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )

        print(f"      Saved to: {output_path}")
        self.report["results"]["int8"] = {
            "path": str(output_path),
            "size_mb": output_path.stat().st_size / (1024 * 1024),
        }
        return output_path

    def validate(
        self, model_paths: dict[str, Path], torch_model: torch.nn.Module
    ) -> None:
        """Validate ONNX models against PyTorch baseline."""
        if not self.args.val_dir:
            print("\n[5/5] Skipping validation (no validation directory provided)")
            return

        print("\n[5/5] Validating models...")
        val_images = sorted(Path(self.args.val_dir).glob("*"))
        val_images = [
            p
            for p in val_images
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ][: self.args.val_count]

        if not val_images:
            print("      No validation images found.")
            return

        results = {k: [] for k in model_paths.keys()}
        results["pytorch"] = []

        print(f"      Testing on {len(val_images)} images...")

        # Pre-calculate PyTorch results to save VRAM
        pt_results = []
        print("      Generating PyTorch baseline...")
        for img_path in val_images:
            input_tensor = preprocess_image(img_path, None, self.args.norm)
            with torch.no_grad():
                pt_input = torch.from_numpy(input_tensor).to(self.device)
                pt_output = torch_model(pt_input).cpu().numpy()
                pt_img = postprocess_output(pt_output, self.args.norm)
                pt_results.append((img_path, input_tensor, pt_img))

        # Clear PyTorch model to free VRAM for ONNX Runtime
        del torch_model
        torch.cuda.empty_cache()

        # Validate ONNX models sequentially to avoid OOM
        for name, path in model_paths.items():
            print(f"      Validating {name}...")
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.args.device == "cuda"
                else ["CPUExecutionProvider"]
            )

            try:
                sess = ort.InferenceSession(str(path), providers=providers)
            except Exception as e:
                print(f"      Error loading {name} model: {e}")
                print(f"      Skipping validation for {name}.")
                continue

            for i, (img_path, input_tensor, pt_img) in enumerate(pt_results):
                try:
                    onnx_output = sess.run(None, {"input": input_tensor})[0]
                    onnx_img = postprocess_output(onnx_output, self.args.norm)

                    # Calculate PSNR vs PyTorch
                    psnr = calculate_psnr(pt_img, onnx_img, border=self.args.scale + 2)
                    results[name].append(psnr)
                except Exception as e:
                    print(f"        Error validating image {i + 1}: {e}")

            # Clean up session
            del sess
            import gc

            gc.collect()

        # Print Summary
        print("\n      Validation Results (PSNR vs PyTorch):")
        print("      -------------------------------------")
        for name, psnrs in results.items():
            if name == "pytorch":
                continue

            if not psnrs:
                print(f"      {name.upper():<5}: Skipped (No results)")
                continue

            avg_psnr = sum(psnrs) / len(psnrs)
            min_psnr = min(psnrs)
            print(
                f"      {name.upper():<5}: Avg: {avg_psnr:.2f} dB | Min: {min_psnr:.2f} dB"
            )
            if name in self.report["results"]:
                self.report["results"][name]["validation_psnr"] = avg_psnr

    def save_report(self) -> None:
        report_path = self.output_dir / "conversion_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2)
        print(f"\nReport saved to {report_path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="ParagonSR2 Release ONNX Converter")

    # Model args
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.safetensors or .pth)",
    )
    parser.add_argument(
        "--arch", required=True, help="Architecture name (e.g. paragonsr2_static_s)"
    )
    parser.add_argument("--scale", type=int, default=2, help="Upscaling factor")
    parser.add_argument(
        "--norm",
        type=str,
        default="01",
        choices=["01", "m11"],
        help="Normalization: '01'=[0,1], 'm11'=[-1,1]",
    )

    # Output args
    parser.add_argument("--output", default="release_onnx", help="Output directory")
    parser.add_argument(
        "--device", default="cuda", help="Device to use for export (cpu/cuda)"
    )

    # INT8 Calibration args
    parser.add_argument(
        "--calib_dir", help="Directory with images for INT8 calibration"
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=256,
        help="Image size for calibration (square)",
    )
    parser.add_argument(
        "--calib_count",
        type=int,
        default=200,
        help="Number of images to use for calibration",
    )

    # Validation args
    parser.add_argument("--val_dir", help="Directory with images for validation")
    parser.add_argument(
        "--val_count", type=int, default=10, help="Number of images to validate"
    )

    args = parser.parse_args()

    converter = ParagonConverter(args)

    # 1. Load PyTorch Model
    torch_model = converter.load_model()

    # 2. Export FP32
    fp32_path = converter.export_fp32(torch_model)

    # 3. Convert FP16
    fp16_path = converter.convert_fp16(fp32_path)

    # 4. Convert INT8
    int8_path = converter.convert_int8(fp32_path)

    # 5. Validate
    models_to_test = {"fp32": fp32_path, "fp16": fp16_path}
    if int8_path:
        models_to_test["int8"] = int8_path

    converter.validate(models_to_test, torch_model)

    converter.save_report()
    print("\nDone.")


if __name__ == "__main__":
    main()
