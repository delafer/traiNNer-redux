#!/usr/bin/env python3
"""
ParagonSR2 Release Converter
============================

Exports ParagonSR2 models to TensorRT-compatible ONNX.

Features:
- Auto-patches AdaptiveAvgPool2d -> ReduceMean (TensorRT Friendly)
- Exports Dynamic FP32 ONNX (Best for trtexec --fp16)
- Validates PSNR match between PyTorch and ONNX
- Embeds weights into a single .onnx file (no .data sidecars)

Usage:
    python convert_onnx_release.py \
        --checkpoint "models/paragon_pro_x4.safetensors" \
        --arch paragonsr2_pro \
        --scale 4 \
        --output "release_output" \
        --device cuda

    # Then build TRT engine:
    trtexec --onnx=release_output/paragonsr2_pro_fp32.onnx \
            --saveEngine=paragonsr2_pro_fp16.trt \
            --fp16 \
            --minShapes=input:1x3x64x64 \
            --optShapes=input:1x3x720x1280 \
            --maxShapes=input:1x3x1080x1920
"""

import argparse
import json
import math
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import shape_inference
from PIL import Image
from torch import nn

# ---------------------------------------------------------------------
# SETUP: Import Architecture
# ---------------------------------------------------------------------
try:
    # Try importing from local directory first (standalone usage)
    sys.path.insert(0, str(Path(__file__).parent))
    import paragonsr2_arch
    from paragonsr2_arch import ParagonSR2

    # Mock registry for standalone script
    ARCH_MAP = {
        "paragonsr2_realtime": paragonsr2_arch.paragonsr2_realtime,
        "paragonsr2_stream": paragonsr2_arch.paragonsr2_stream,
        "paragonsr2_photo": paragonsr2_arch.paragonsr2_photo,
    }

except ImportError:
    # Fallback to traiNNer repo structure
    try:
        repo_root = Path(__file__).parents[2]
        sys.path.insert(0, str(repo_root))
        import traiNNer.archs.paragonsr2_arch  # Import to register architectures
        from traiNNer.utils.registry import ARCH_REGISTRY

        ARCH_MAP = ARCH_REGISTRY
    except ImportError as e:
        print("CRITICAL ERROR: Could not import architecture definition.")
        print(
            "Ensure 'paragonsr2_arch.py' is in the same folder or traiNNer is installed."
        )
        sys.exit(1)

warnings.filterwarnings("ignore")


# =============================================================================
# HELPER: TensorRT Compatibility Patcher
# =============================================================================


class TensorRTGlobalAvgPool(nn.Module):
    """
    Replaces nn.AdaptiveAvgPool2d(1) with torch.mean().
    Result: ONNX 'GlobalAveragePool' or 'ReduceMean' (TRT Optimized).
    """

    def forward(self, x):
        return x.mean(dim=[-1, -2], keepdim=True)


def patch_model_for_tensorrt(model: nn.Module):
    """Recursively replace AdaptiveAvgPool2d(1) with TensorRTGlobalAvgPool."""
    print("      Patching model for TensorRT compatibility...")
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            # Check if output size is 1x1
            out = module.output_size
            is_one = (out == 1) if isinstance(out, int) else (out == (1, 1))

            if is_one:
                # Replace logic
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                setattr(parent, child_name, TensorRTGlobalAvgPool())
                replaced_count += 1

    print(f"      Replaced {replaced_count} AdaptiveAvgPool layers.")
    return model


# =============================================================================
# UTILITIES
# =============================================================================


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
    if border > 0:
        img1 = img1[border:-border, border:-border, :]
        img2 = img2[border:-border, border:-border, :]
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 1e-10:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def preprocess_image(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img_array, axis=0)  # -> BCHW


def postprocess_output(output: np.ndarray) -> np.ndarray:
    output = output.squeeze(0)
    output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
    output = np.clip(output, 0, 1)
    return (output * 255.0).round().astype(np.uint8)


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
        print(f"\n[1/4] Loading architecture: {self.args.arch}")
        # Try to get from registry or map
        arch_fn = ARCH_MAP.get(self.args.arch)

        if arch_fn is None:
            raise ValueError(f"Architecture '{self.args.arch}' not found in registry.")

        # Build architecture kwargs from command line overrides
        arch_kwargs = {"scale": self.args.scale}

        if self.args.use_content_aware is not None:
            arch_kwargs["use_content_aware"] = self.args.use_content_aware
            print(f"      Override: use_content_aware={self.args.use_content_aware}")

        if self.args.upsampler_alpha is not None:
            arch_kwargs["upsampler_alpha"] = self.args.upsampler_alpha
            print(f"      Override: upsampler_alpha={self.args.upsampler_alpha}")

        if self.args.detail_gain is not None:
            arch_kwargs["detail_gain"] = self.args.detail_gain
            print(f"      Override: detail_gain={self.args.detail_gain}")

        if self.args.attention_mode is not None:
            arch_kwargs["attention_mode"] = self.args.attention_mode
            print(f"      Override: attention_mode={self.args.attention_mode}")

        if self.args.export_safe is not None:
            arch_kwargs["export_safe"] = self.args.export_safe
            print(f"      Override: export_safe={self.args.export_safe}")

        if self.args.window_size is not None:
            arch_kwargs["window_size"] = self.args.window_size
            print(f"      Override: window_size={self.args.window_size}")

        model = arch_fn(**arch_kwargs)

        print(f"      Loading weights: {self.args.checkpoint}")
        if str(self.args.checkpoint).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(self.args.checkpoint)
        else:
            state_dict = torch.load(self.args.checkpoint, map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]

        # Handle 'module.' prefix if from DDP
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model.load_state_dict(new_dict, strict=True)
        model.to(self.device).eval()

        # Optimization hooks
        if hasattr(model, "fuse_for_release"):
            print("      Fusing release blocks...")
            model.fuse_for_release()

        # Patch for TRT
        model = patch_model_for_tensorrt(model)
        return model

    def export_fp32(self, model: torch.nn.Module) -> Path:
        print("\n[2/4] Exporting FP32 ONNX...")
        output_path = self.output_dir / f"{self.args.arch}_fp32.onnx"

        # Dummy Input (Standard HD size ensures no weird shape inference issues)
        dummy_input = torch.randn(1, 3, 64, 64, device=self.device)

        # Export
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=18,  # Updated for PyTorch 2.5+ / TRT 8.6+
            do_constant_folding=True,
            # For future Custom Symbolic / FlexAttention export:
            # custom_opsets={"com.custom": 1},
        )

        # Optimizing / Cleaning
        print("      Simplifying ONNX graph...")
        try:
            onnx_model = onnx.load(str(output_path))

            # Infer shapes
            try:
                onnx_model = shape_inference.infer_shapes(onnx_model)
            except Exception as e:
                print(f"      Warning: Shape inference failed ({e}), skipping.")

            # Save cleaned model (forcing < 2GB constraint logic if needed, but here usually small)
            onnx.save(onnx_model, str(output_path))

            # Clean external data files if accidentally created
            data_file = Path(str(output_path) + ".data")
            if data_file.exists():
                data_file.unlink()

        except Exception as e:
            print(f"      Warning: ONNX simplification failed: {e}")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"      Success: {output_path.name} ({size_mb:.2f} MB)")
        return output_path

    def validate(self, model_path: Path, torch_model: torch.nn.Module) -> None:
        if not self.args.val_dir:
            return

        print("\n[3/4] Validating Accuracy (PyTorch vs ONNX)...")
        val_dir = Path(self.args.val_dir)
        val_images = sorted(
            [
                p
                for p in val_dir.glob("*")
                if p.suffix.lower() in [".png", ".jpg", ".webp"]
            ]
        )[: self.args.val_count]

        if not val_images:
            print("      No images found for validation.")
            return

        print(f"      Testing on {len(val_images)} images...")

        # Set up ONNX Runtime
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.args.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        try:
            sess = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as e:
            print(f"      Error loading ONNX Runtime: {e}")
            return

        psnrs = []

        for img_path in val_images:
            # Prepare Input
            inp = preprocess_image(img_path)

            # PyTorch Inference
            with torch.no_grad():
                pt_in = torch.from_numpy(inp).to(self.device)
                pt_out = torch_model(pt_in).cpu().numpy()
                pt_img = postprocess_output(pt_out)

            # ONNX Inference
            onnx_out = sess.run(None, {"input": inp})[0]
            onnx_img = postprocess_output(onnx_out)

            # Compare
            # Ignore border pixels (scale + 2) to avoid padding differences
            score = calculate_psnr(pt_img, onnx_img, border=self.args.scale + 4)
            psnrs.append(score)
            # print(f"      {img_path.name}: {score:.2f} dB")

        avg_psnr = sum(psnrs) / len(psnrs)
        status = "✅ PASS" if avg_psnr > 50 else "⚠️  FAIL"
        print(f"      Average PSNR Match: {avg_psnr:.2f} dB  ->  {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParagonSR2 Release Converter")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pth or .safetensors"
    )
    parser.add_argument(
        "--arch", required=True, help="Model variant (e.g., paragonsr2_photo)"
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output", default="release_onnx", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument("--val_dir", help="Folder of images to validate ONNX accuracy")
    parser.add_argument(
        "--val_count", type=int, default=5, help="Number of images to test"
    )

    # Architecture override arguments (for checkpoints trained with non-default settings)
    parser.add_argument(
        "--use_content_aware",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Override use_content_aware setting (true/false)",
    )
    parser.add_argument(
        "--upsampler_alpha",
        type=float,
        default=None,
        help="Override upsampler_alpha setting",
    )
    parser.add_argument(
        "--detail_gain",
        type=float,
        default=None,
        help="Override detail_gain setting",
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        default=None,
        help="Override attention_mode (sdpa, flex, none)",
    )
    parser.add_argument(
        "--export_safe",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Override export_safe setting (true/false)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Override window_size",
    )

    args = parser.parse_args()

    converter = ParagonConverter(args)

    # 1. Load PyTorch Model
    torch_model = converter.load_model()

    # 2. Export ONNX
    onnx_path = converter.export_fp32(torch_model)

    # 3. Validate
    if args.val_dir:
        converter.validate(onnx_path, torch_model)

    print("\n" + "=" * 60)
    print("DONE. Model ready for TensorRT conversion.")
    print(f"File: {onnx_path}")
    print("=" * 60)
