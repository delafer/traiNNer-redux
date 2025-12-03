#!/usr/bin/env python3
"""
ParagonSR2 Release Converter (TensorRT Patched & Clean)
=======================================================

- Opset 18 (Native PyTorch 2.x support)
- TensorRT Patch (Replaces AdaptiveAvgPool with ReduceMean)
- Single File Output (Merged .data)

Usage:
    python scripts/paragonsr2/convert_onnx_release.py \
        --checkpoint models/my_model.safetensors \
        --arch paragonsr2_static_s \
        --scale 2 \
        --output release_output

And then

trtexec \
  --onnx=release_models/2xParagonSR2_Nano_fidelity/paragonsr2_nano_fp32.onnx \
  --saveEngine=release_models/2xParagonSR2_Nano_fidelity/paragonsr2_nano_2x_fp16.trt \
  --fp16 \
  --minShapes=input:1x3x64x64 \
  --optShapes=input:1x3x540x960 \
  --maxShapes=input:1x3x1080x1920

And then

python scripts/paragonsr2/inference_trt.py \
  --engine release_models/2xParagonSR2_Nano_fidelity/paragonsr2_nano_2x_fp16.trt \
  --input /home/phips/Documents/dataset/cc0/val_lr_x2 \
  --output /home/phips/Documents/dataset/cc0/val_lr__x2_trt \
  --scale 2
"""

import argparse
import json
import math
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

# Import architectures
try:
    # Add repo root to path
    repo_root = Path(__file__).parents[2]
    sys.path.insert(0, str(repo_root))

    import traiNNer.archs.paragonsr2_arch  # Main ParagonSR2 architecture
    from traiNNer.utils.registry import ARCH_REGISTRY
except ImportError as e:
    print(f"Error: Could not import traiNNer modules: {e}")
    print(f"Repo root: {Path(__file__).parents[2]}")
    print(f"sys.path: {sys.path[:3]}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =============================================================================
# HELPER: TensorRT Compatibility Patcher
# =============================================================================


class TensorRTGlobalAvgPool(nn.Module):
    """
    Replaces nn.AdaptiveAvgPool2d(1).
    Uses torch.mean() which exports to ONNX 'ReduceMean'.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.mean(dim=[-1, -2], keepdim=True)


def patch_model_for_tensorrt(model: nn.Module):
    """Recursively replace AdaptiveAvgPool2d(1) with TensorRTGlobalAvgPool."""
    print("      Patching model for TensorRT compatibility...")
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            output_size = module.output_size
            if isinstance(output_size, int):
                is_one = output_size == 1
            else:
                is_one = output_size == (1, 1)

            if is_one:
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                setattr(parent, child_name, TensorRTGlobalAvgPool())
                replaced_count += 1
    print(f"      Replaced {replaced_count} AdaptiveAvgPool layers with TRT-safe Mean.")
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


def preprocess_image(image_path: Path, norm_type: str = "01") -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    if norm_type == "m11":
        img_array = (img_array - 0.5) / 0.5
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0)


def postprocess_output(output: np.ndarray, norm_type: str = "01") -> np.ndarray:
    output = output.squeeze(0)
    output = np.transpose(output, (1, 2, 0))
    if norm_type == "m11":
        output = (output * 0.5) + 0.5
    output = np.clip(output, 0, 1)
    return (output * 255.0).round().astype(np.uint8)


# =============================================================================
# CONVERTER
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
        print(f"\n[1/4] Loading model: {self.args.arch}")
        arch_fn = ARCH_REGISTRY.get(self.args.arch)
        model = arch_fn(scale=self.args.scale)

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
        model.to(self.device).eval()

        if hasattr(model, "fuse_for_release"):
            print("      Fusing ReparamConvV2 blocks...")
            model.fuse_for_release()

        model = patch_model_for_tensorrt(model)
        return model

    def export_fp32(self, model: torch.nn.Module) -> Path:
        print("\n[2/4] Exporting FP32 ONNX...")
        output_path = self.output_dir / f"{self.args.arch}_fp32.onnx"
        dummy_input = torch.randn(1, 3, 64, 64, device=self.device)

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
            opset_version=18,  # <--- FIXED: Explicitly set to 18
            do_constant_folding=True,
        )

        try:
            onnx_model = onnx.load(str(output_path))
            try:
                onnx_model = shape_inference.infer_shapes(onnx_model)
            except:
                pass

            # Merge .data file into .onnx
            onnx.save(onnx_model, str(output_path), save_as_external_data=False)

            # Clean up leftover
            data_file = Path(str(output_path) + ".data")
            if data_file.exists():
                data_file.unlink()

        except Exception as e:
            print(f"      Packing error: {e}")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"      Saved FP32: {output_path} ({size_mb:.2f} MB)")
        self.report["results"]["fp32"] = {"path": str(output_path), "size_mb": size_mb}
        return output_path

    def validate(
        self, model_paths: dict[str, Path], torch_model: torch.nn.Module
    ) -> None:
        if not self.args.val_dir:
            return
        print("\n[4/4] Validating Models...")

        val_images = sorted(
            [
                p
                for p in Path(self.args.val_dir).glob("*")
                if p.suffix.lower() in [".png", ".jpg", ".webp"]
            ]
        )[: self.args.val_count]
        if not val_images:
            return

        print("      Generating PyTorch baseline...")
        pt_results = []
        for img_path in val_images:
            inp = preprocess_image(img_path, self.args.norm)
            with torch.no_grad():
                pt_tensor = torch.from_numpy(inp).to(self.device)
                pt_out = torch_model(pt_tensor).cpu().numpy()
                pt_img = postprocess_output(pt_out, self.args.norm)
                pt_results.append((img_path, inp, pt_img))

        del torch_model
        torch.cuda.empty_cache()

        print(f"\n      {'Model':<10} | {'Avg PSNR':<10} | {'Status':<15}")
        print("      " + "-" * 45)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.args.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        for name, path in model_paths.items():
            if path is None:
                continue
            try:
                sess = ort.InferenceSession(str(path), providers=providers)
                psnrs = []
                for _i, (_, inp, pt_img) in enumerate(pt_results):
                    onnx_out = sess.run(None, {"input": inp})[0]
                    onnx_img = postprocess_output(onnx_out, self.args.norm)
                    psnrs.append(
                        calculate_psnr(pt_img, onnx_img, border=self.args.scale + 2)
                    )
                avg = sum(psnrs) / len(psnrs)
                status = "✅ Excellent" if avg > 50 else "⚠️ Degradation"
                print(f"      {name.upper():<10} | {avg:<10.2f} | {status:<15}")
                self.report["results"][name]["validation_psnr"] = avg
            except Exception as e:
                print(f"      {name.upper():<10} | {'FAILED':<10} | {str(e)[:30]}...")

    def save_report(self) -> None:
        with open(self.output_dir / "conversion_report.json", "w") as f:
            json.dump(self.report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--norm", default="01")
    parser.add_argument("--output", default="release_onnx")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_dir")
    parser.add_argument("--val_count", type=int, default=10)

    args = parser.parse_args()
    converter = ParagonConverter(args)
    torch_model = converter.load_model()

    fp32_path = converter.export_fp32(torch_model)
    converter.validate({"fp32": fp32_path}, torch_model)
    converter.save_report()

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Your ONNX model is located at: {fp32_path}")
    print("Run the trtexec command again.")
    print("=" * 60)
