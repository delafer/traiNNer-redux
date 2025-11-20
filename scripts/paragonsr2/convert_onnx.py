#!/usr/bin/env python3
"""
Universal ParagonSR2 export tool:
- Loads checkpoint from registry
- Fuses model for deployment
- Exports ONNX FP32 with opset fallback
- Exports ONNX FP16 with intelligent layer handling
- Generates INT8 ONNX using calibration folder
- Validates numerical differences against PyTorch model
- Supports dynamic shapes and GPU export
- Smart FP16 conversion (TensorRT-like behavior)

Usage:
    python convert_onnx.py \
        --checkpoint ./experiments/2xParagonSR2_static_micro_400k.safetensors \
        --arch paragonsr2_static_micro \
        --scale 2 \
        --output ./export_output \
        --calib ./dataset/hr \
        --val ./dataset/val_lr \
        --device cuda \
        --verbose

python scripts/paragonsr2/convert_onnx.py \
    --checkpoint ./experiments/net_g_ema_75000.safetensors \
    --arch paragonsr2_static_micro \
    --scale 2 \
    --output ./export \
    --device cuda \
    --calib ./dataset/hr \
    --val ./dataset/val_lr \
    --input_size 256 256 \
    --samples 20 \
    --verbose


Author: Philip Hofmann (traiNNer-redux)
License: MIT
"""

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch
import traiNNer.archs.paragonsr2_static_arch  # Ensure architectures are registered
from onnx import checker, helper
from onnxruntime.quantization import QuantType, quantize_static
from PIL import Image

# Architecture imports for registry
from traiNNer.utils.registry import ARCH_REGISTRY

# Optional imports
try:
    from onnxconverter_common import float16

    HAS_ONNXTENSORRT = True
except ImportError:
    HAS_ONNXTENSORRT = False

try:
    import tensorrt as trt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

# Smart FP16 conversion - operators that should stay in FP32 for stability
FP16_CRITICAL_OPS = {
    "GroupNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "ReduceMean",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "Softmax",
    "LogSoftmax",
    "Sigmoid",
    "Tanh",
    "BatchNormalization",
    "Dropout",
    "TopK",
}

# FP16 safe operations (don't need special handling)
FP16_SAFE_OPS = {
    "Conv",
    "ConvTranspose",
    "MatMul",
    "Gemm",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Relu",
    "LeakyRelu",
    "PReLU",
    "ReLU6",
    "Elu",
    "Selu",
    "AveragePool",
    "MaxPool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Resize",
    "Upsample",
    "Concat",
    "Split",
    "Slice",
    "Pad",
    "Tile",
    "Transpose",
    "Reshape",
    "Flatten",
    "Expand",
    "Broadcast",
    "Identity",
    "Cast",
    "Shape",
    "Gather",
    "Scatter",
    "Where",
    "Clip",
    "Floor",
    "Ceil",
    "Round",
    "Sqrt",
    "Pow",
    "Exp",
    "Log",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Atanh",
    "Asinh",
    "Acosh",
}


def psnr_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute PSNR between two numpy arrays."""
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


class ImageFolderCalibration:
    """Calibration dataloader for INT8 quantization."""

    def __init__(
        self, folder: str, input_name: str, input_size: tuple = (128, 128)
    ) -> None:
        self.images = list(Path(folder).glob("*"))
        self.input_name = input_name
        self.index = 0
        self.input_size = input_size

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Get next calibration image."""
        if self.index >= len(self.images):
            return None

        p = self.images[self.index]
        self.index += 1

        img = (
            Image.open(p)
            .convert("RGB")
            .resize(self.input_size[::-1], Image.Resampling.BICUBIC)
        )
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None]
        return {self.input_name: arr}


def validate_model_compatibility(onnx_path: Path) -> dict[str, Any]:
    """Analyze ONNX model for deployment compatibility."""
    try:
        model = onnx.load(onnx_path)
        checker.check_model(model)

        compatibility = {
            "valid": True,
            "opsets": [opset.version for opset in model.opset_import],
            "total_nodes": len(model.graph.node),
            "has_dynamic_axes": any(
                dim.type.HasField("dim_param")
                for input in model.graph.input
                for dim in input.type.tensor_type.shape.dim
            ),
            "problematic_ops": [],
            "fp16_compatible": True,
        }

        # Check for problematic ops
        for node in model.graph.node:
            op_type = node.op_type

            if op_type in FP16_CRITICAL_OPS:
                compatibility["problematic_ops"].append(op_type)
                if op_type in {
                    "GroupNormalization",
                    "LayerNormalization",
                    "InstanceNormalization",
                }:
                    compatibility["fp16_compatible"] = False

        if compatibility["problematic_ops"]:
            print(f"⚠️  Found FP16-incompatible ops: {compatibility['problematic_ops']}")
        else:
            print("✅ Model appears fully FP16 compatible")

        return compatibility

    except Exception as e:
        return {"valid": False, "error": str(e)}


def compare_pytorch_and_onnx(
    pt_model,
    onnx_path: Path,
    samples: list[Path],
    device: str = "cpu",
    input_size: tuple = (128, 128),
) -> dict[str, float]:
    """Compare PyTorch and ONNX outputs with detailed analysis."""
    print(f"\n🔍 [VALIDATION] Testing {onnx_path.name}")

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    pt_model.eval()
    pt_model.to(device)

    psnrs = []
    max_diff = []

    with torch.no_grad():
        for i, p in enumerate(samples):
            img = (
                Image.open(p)
                .convert("RGB")
                .resize((input_size[1], input_size[0]), Image.Resampling.BICUBIC)
            )
            arr = np.asarray(img).astype(np.float32) / 255.0
            inp = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None]).to(device)

            # PyTorch output
            out_pt = pt_model(inp).cpu().numpy()[0].transpose(1, 2, 0)

            # ONNX output
            ort_in = {input_name: inp.cpu().numpy()}
            out_onnx = sess.run(None, ort_in)[0][0].transpose(1, 2, 0)

            # Metrics
            ps = psnr_numpy(out_pt.clip(0, 1), out_onnx.clip(0, 1))
            psnrs.append(ps)

            diff = np.abs(out_pt - out_onnx)
            max_diff.append(np.max(diff))

            if i < 3:  # Show first few comparisons
                print(f"  Sample {i + 1}: PSNR={ps:.2f}dB, MaxDiff={np.max(diff):.6f}")

    results = {
        "mean_psnr": np.mean(psnrs),
        "min_psnr": np.min(psnrs),
        "max_psnr": np.max(psnrs),
        "mean_max_diff": np.mean(max_diff),
        "max_max_diff": np.max(max_diff),
    }

    print(
        f"  📊 Summary: Mean PSNR={results['mean_psnr']:.2f}dB, "
        f"MaxDiff={results['max_max_diff']:.6f}"
    )

    return results


def smart_fp16_conversion(fp32_path: Path, fp16_path: Path, output_dir: Path) -> bool:
    """Convert FP32 ONNX to FP16 with intelligent operator handling."""
    print(f"\n🎯 [FP16] Converting {fp32_path.name} to FP16...")

    try:
        # Load FP32 model
        model = onnx.load(fp32_path)

        # Analyze compatibility
        compatibility = validate_model_compatibility(fp32_path)
        if not compatibility["valid"]:
            print(f"❌ FP32 model validation failed: {compatibility.get('error')}")
            return False

        # Smart conversion with op blocklist
        if compatibility["problematic_ops"]:
            # Convert to FP16 but keep problematic ops in FP32
            blocklist = list(set(compatibility["problematic_ops"]))
            print(f"🔧 Keeping FP32: {blocklist}")

            model_fp16 = float16.convert_float_to_float16(
                model, keep_io_types=True, op_block_list=blocklist
            )
        else:
            # Full FP16 conversion
            print("🔧 Full FP16 conversion")
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

        # Save FP16 model
        onnx.save(model_fp16, fp16_path)

        # Validate FP16 model
        checker.check_model(fp16_path)
        print(f"✅ FP16 conversion successful: {fp16_path}")

        # Generate report
        report_path = output_dir / "fp16_conversion_report.txt"
        with open(report_path, "w") as f:
            f.write("FP16 Conversion Report\n")
            f.write("=====================\n\n")
            f.write(f"Source: {fp32_path.name}\n")
            f.write(f"Output: {fp16_path.name}\n\n")
            f.write(f"Original opsets: {compatibility['opsets']}\n")
            f.write(f"Problematic ops: {compatibility['problematic_ops']}\n")
            f.write(f"FP16 compatible: {compatibility['fp16_compatible']}\n")
            f.write(f"Total nodes: {compatibility['total_nodes']}\n")
            f.write(f"Has dynamic axes: {compatibility['has_dynamic_axes']}\n")

        return True

    except Exception as e:
        print(f"❌ FP16 conversion failed: {e}")
        return False


def export_tensorrt_engine(onnx_path: Path, output_dir: Path) -> bool:
    """Export TensorRT engine from ONNX (if TensorRT available)."""
    if not HAS_TENSORRT:
        print("ℹ️  TensorRT not available, skipping engine export")
        return False

    print(f"\n🚀 [TENSORRT] Building engine from {onnx_path.name}...")

    try:
        # Create TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)

        # Build engine
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)

        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("❌ ONNX parsing failed")
                return False

        # Build engine
        engine = builder.build_engine(network, config)

        if engine:
            # Save engine
            engine_path = output_dir / f"{onnx_path.stem}.trt"
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())

            print(f"✅ TensorRT engine saved: {engine_path}")

            # Report layers
            layers = []
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.precision_is_set():
                    layers.append(f"Layer {i}: {layer.name} @ {layer.precision}")

            if layers:
                report_path = output_dir / "tensorrt_precision_report.txt"
                with open(report_path, "w") as f:
                    f.write("TensorRT Precision Report\n")
                    f.write("========================\n\n")
                    for layer in layers:
                        f.write(layer + "\n")
                print(f"📊 Precision report saved: {report_path}")

            return True
        else:
            print("❌ Engine building failed")
            return False

    except Exception as e:
        print(f"❌ TensorRT export failed: {e}")
        return False


def export_everything() -> None:
    """Main export function."""
    parser = argparse.ArgumentParser(description="Universal ParagonSR2 export tool")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--arch", required=True, help="Architecture name (e.g., paragonsr2_static_s)"
    )
    parser.add_argument("--output", required=True, help="Output directory")

    # Optional arguments
    parser.add_argument("--scale", type=int, default=4, help="SR scale factor")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device for export"
    )
    parser.add_argument("--calib", help="Folder for INT8 calibration images")
    parser.add_argument("--val", help="Folder for validation images")
    parser.add_argument(
        "--input_size",
        nargs=2,
        type=int,
        default=[128, 128],
        help="Input image size (height width)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of validation samples"
    )

    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🎯 ParagonSR2 Universal Export Tool")
    print("===================================")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {args.arch}")
    print(f"Scale: {args.scale}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Check required imports
    if not HAS_ONNXTENSORRT:
        print("⚠️  onnxconverter_common not available - FP16 conversion limited")

    if not HAS_TENSORRT:
        print("ℹ️  TensorRT not available - engine export disabled")

    # Load model from registry
    print("\n📦 Loading model...")
    try:
        arch_fn = ARCH_REGISTRY.get(args.arch)
        if arch_fn is None:
            available = list(ARCH_REGISTRY._obj_map.keys())
            raise ValueError(
                f"Architecture '{args.arch}' not found. Available: {available}"
            )

        model = arch_fn(scale=args.scale)

        # Load checkpoint (handle both .pth and .safetensors)
        if args.checkpoint.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(args.checkpoint, device="cpu")
            except ImportError:
                raise ImportError(
                    "safetensors not installed. Install with: pip install safetensors"
                )
        else:
            # Use weights_only=False for PyTorch 2.6 compatibility
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))

        model.load_state_dict(state_dict, strict=True)

        print(f"✅ Model loaded: {args.arch} (scale={args.scale})")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # Fuse for deployment
    if hasattr(model, "fuse_for_release"):
        print("\n🔧 Fusing model for deployment...")
        model.fuse_for_release()
        print("✅ Model fused")

    # Prepare for export
    model.eval()
    model.to(args.device)

    # Create dummy input
    dummy = torch.randn(1, 3, args.input_size[0], args.input_size[1]).to(args.device)

    # Dynamic axes for flexible input sizes
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    # Export FP32
    print("\n📊 Exporting FP32 ONNX...")
    fp32_path = output_dir / "model_fp32.onnx"

    # Try different opsets if needed
    opsets_to_try = [args.opset, 16, 15, 14]
    fp32_success = False

    for opset in opsets_to_try:
        try:
            if args.verbose:
                print(f"  Trying opset {opset}...")

            torch.onnx.export(
                model,
                dummy,
                fp32_path,
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
            )

            checker.check_model(fp32_path)
            print(f"✅ FP32 export successful (opset {opset}): {fp32_path}")
            fp32_success = True
            break

        except Exception as e:
            if args.verbose:
                print(f"  ❌ Failed with opset {opset}: {e}")
            continue

    if not fp32_success:
        print("❌ All FP32 export attempts failed")
        return

    # Export FP16 (if onnxconverter_common available)
    if HAS_ONNXTENSORRT:
        print("\n🎯 Exporting FP16 ONNX...")
        fp16_path = output_dir / "model_fp16.onnx"

        # Try direct export first
        try:
            if args.verbose:
                print("  Trying direct half() export...")

            model_fp16 = model.half().to(args.device)
            torch.onnx.export(
                model_fp16,
                dummy.half(),
                fp16_path,
                opset_version=args.opset,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )

            checker.check_model(fp16_path)
            print(f"✅ Direct FP16 export successful: {fp16_path}")

        except Exception as e:
            if args.verbose:
                print(f"  Direct export failed: {e}")

            # Fallback to smart conversion
            fp16_success = smart_fp16_conversion(fp32_path, fp16_path, output_dir)
            if not fp16_success:
                print("❌ FP16 conversion failed")
    else:
        print("\n⚠️  FP16 export skipped (onnxconverter_common not available)")

    # Export INT8 (if calibration data available)
    if args.calib and args.calib != "":
        print("\n🔢 Exporting INT8 ONNX...")
        try:
            sess = ort.InferenceSession(str(fp32_path))
            input_name = sess.get_inputs()[0].name

            calib_reader = ImageFolderCalibration(
                args.calib, input_name, tuple(args.input_size)
            )
            int8_path = output_dir / "model_int8.onnx"

            quantize_static(
                str(fp32_path), str(int8_path), calib_reader, quant_format=QuantType.QDQ
            )
            print(f"✅ INT8 export successful: {int8_path}")

        except Exception as e:
            print(f"❌ INT8 export failed: {e}")
    else:
        print("\nℹ️  INT8 export skipped (no calibration data)")

    # TensorRT engine export (if available)
    if HAS_TENSORRT:
        if (output_dir / "model_fp16.onnx").exists():
            export_tensorrt_engine(output_dir / "model_fp16.onnx", output_dir)
        elif (output_dir / "model_fp32.onnx").exists():
            export_tensorrt_engine(output_dir / "model_fp32.onnx", output_dir)

    # Validation
    if args.val and Path(args.val).exists():
        print("\n🔍 Running validation...")
        val_imgs = list(Path(args.val).glob("*"))[: args.samples]

        if not val_imgs:
            print("⚠️  No validation images found")
        else:
            print(f"  Testing {len(val_imgs)} samples...")

            # Test all exported formats
            formats_to_test = ["model_fp32.onnx", "model_fp16.onnx", "model_int8.onnx"]

            for fmt in formats_to_test:
                fmt_path = output_dir / fmt
                if fmt_path.exists():
                    results = compare_pytorch_and_onnx(
                        model, fmt_path, val_imgs, args.device, tuple(args.input_size)
                    )

                    # Save results
                    results_path = output_dir / f"{fmt_path.stem}_validation.json"
                    import json

                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)

    # Summary
    print("\n🎉 Export Complete!")
    print("==================")
    print(f"Output directory: {output_dir}")

    # List generated files
    exported_files = list(output_dir.glob("*.onnx")) + list(output_dir.glob("*.trt"))
    if exported_files:
        print("Generated files:")
        for f in sorted(exported_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  📄 {f.name} ({size_mb:.1f} MB)")

    # Generate deployment guide
    guide_path = output_dir / "deployment_guide.md"
    with open(guide_path, "w") as f:
        f.write("# ParagonSR2 Deployment Guide\n\n")
        f.write("## Generated Files\n\n")
        f.write("| File | Precision | Use Case | Size |\n")
        f.write("|------|-----------|----------|------|\n")

        for file_path in sorted(exported_files):
            if file_path.suffix == ".onnx":
                precision = file_path.stem.split("_")[-1].upper()
                use_case = {
                    "fp32": "High accuracy, CPU/backup",
                    "fp16": "Balanced speed/accuracy, GPU",
                    "int8": "Fastest inference, edge deployment",
                }.get(precision, "Custom")
            elif file_path.suffix == ".trt":
                precision = "TensorRT"
                use_case = "Optimized GPU inference"

            size_mb = file_path.stat().st_size / (1024 * 1024)
            f.write(
                f"| {file_path.name} | {precision} | {use_case} | {size_mb:.1f} MB |\n"
            )

        f.write("\n## Usage Examples\n\n")
        f.write("### ONNX Runtime\n\n")
        f.write("```python\n")
        f.write("import onnxruntime as ort\n")
        f.write("session = ort.InferenceSession('model_fp16.onnx')\n")
        f.write("output = session.run(None, {'input': input_data})\n")
        f.write("```\n\n")

        f.write("### TensorRT (Python)\n\n")
        f.write("```python\n")
        f.write("import tensorrt as trt\n")
        f.write("# Load engine and run inference\n")
        f.write("```\n\n")

        f.write("## Performance Tips\n\n")
        f.write("- **FP16**: Use for GPU inference (best speed/accuracy balance)\n")
        f.write("- **INT8**: Use for edge deployment (fastest, requires calibration)\n")
        f.write(
            "- **FP32**: Use for CPU inference or when maximum accuracy is needed\n"
        )

    print(f"📚 Deployment guide: {guide_path}")


if __name__ == "__main__":
    export_everything()
