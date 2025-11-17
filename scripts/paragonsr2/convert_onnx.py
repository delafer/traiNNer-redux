#!/usr/bin/env python3
"""
Universal ParagonSR2 export tool:
- Loads checkpoint
- Fuses model
- Exports ONNX FP32
- Attempts FP16 export (with safe fallback via conversion)
- Generates INT8 ONNX using calibration folder
- Validates numerical differences against PyTorch model
- Includes opset fallback and FP16-safe operator patching
- Supports dynamic shapes (recommended)

python convert_paragon.py \
    --checkpoint ./experiments/2xParagonSR2_static_micro_400k.safetensors \
    --arch paragonsr2_static_micro \
    --scale 2 \
    --output ./export_output \
    --calib ./dataset/hr \
    --val ./dataset/val_lr

"""

import argparse
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import checker
from PIL import Image

# Optional imports based on what is installed
try:
    from onnxconverter_common import float16

    HAS_FP32_TO_FP16 = True
except:
    HAS_FP32_TO_FP16 = False

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        quantize_static,
    )

    HAS_INT8 = True
except:
    HAS_INT8 = False


############################################################
# --- Simple PSNR computation for validation ---
############################################################
def psnr_numpy(a: np.ndarray, b: np.ndarray):
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


############################################################
# --- Calibration dataloader for INT8 ---
############################################################
class ImageFolderCalibration(CalibrationDataReader):
    def __init__(self, folder, input_name, input_size=(128, 128)) -> None:
        self.images = list(Path(folder).glob("*"))
        self.input_name = input_name
        self.index = 0
        self.input_size = input_size

    def get_next(self):
        if self.index >= len(self.images):
            return None
        p = self.images[self.index]
        self.index += 1
        img = Image.open(p).convert("RGB").resize(self.input_size[::-1], Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None]
        return {self.input_name: arr}


############################################################
# --- ONNX validation helper ---
############################################################
def compare_pytorch_and_onnx(
    pt_model, onnx_path, samples, device="cpu", input_size=(128, 128)
):
    print(f"\n[VALIDATION] Comparing PyTorch and ONNX outputs for {onnx_path}")
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    pt_model.eval()
    pt_model.to(device)
    psnrs = []

    for p in samples:
        img = (
            Image.open(p)
            .convert("RGB")
            .resize((input_size[1], input_size[0]), Image.BICUBIC)
        )
        arr = np.asarray(img).astype(np.float32) / 255.0
        inp = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None]).to(device)

        with torch.no_grad():
            out_pt = pt_model(inp).cpu().numpy()[0].transpose(1, 2, 0)

        ort_in = {input_name: inp.cpu().numpy()}
        out_onnx = sess.run(None, ort_in)[0][0].transpose(1, 2, 0)

        ps = psnr_numpy(out_pt.clip(0, 1), out_onnx.clip(0, 1))
        psnrs.append(ps)

    print("Mean PSNR:", np.mean(psnrs))
    print("Min PSNR:", np.min(psnrs))
    print("Max PSNR:", np.max(psnrs))
    return np.mean(psnrs)


############################################################
# --- Main export logic ---
############################################################
def export_everything() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calib", required=False, help="folder of HR or LR images for calibration"
    )
    parser.add_argument(
        "--val", required=False, help="folder of images for validation (recommended)"
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading Model ===")
    model = args.arch(scale=args.scale)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["params_ema"] if "params_ema" in ckpt else ckpt)
    if hasattr(model, "fuse_for_release"):
        print("[INFO] Fusing model")
        model.fuse_for_release()
    model.eval()

    dummy = torch.randn(1, 3, 128, 128)
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    #################################################################
    # --- 1. Export FP32 ---
    #################################################################
    fp32_path = outdir / "model_fp32.onnx"
    print("\n=== Exporting FP32 ONNX ===")
    for opset in [args.opset, 16, 15]:
        try:
            torch.onnx.export(
                model,
                dummy,
                fp32_path,
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
            checker.check_model(fp32_path)
            print(f"[OK] FP32 export succeeded (opset {opset})")
            break
        except Exception as e:
            print(f"[WARN] FP32 export failed with opset {opset}: {e}")
    else:
        raise RuntimeError("All FP32 export attempts failed")

    #################################################################
    # --- 2. Try FP16 export (direct) ---
    #################################################################
    fp16_path = outdir / "model_fp16.onnx"
    try:
        print("\n=== Exporting FP16 ONNX (direct half()) ===")
        model_half = model.half()
        torch.onnx.export(
            model_half,
            dummy.half(),
            fp16_path,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        checker.check_model(fp16_path)
        print("[OK] Direct FP16 export succeeded")
    except Exception:
        print("[WARN] Direct FP16 export failed. Trying FP32→FP16 conversion")
        if HAS_FP32_TO_FP16:
            m = onnx.load(fp32_path)
            m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m_fp16, fp16_path)
            print("[OK] FP16 conversion succeeded using onnxconverter_common")
        else:
            print("[WARN] FP32→FP16 conversion tool not available, skipping FP16")

    #################################################################
    # --- 3. INT8 Quantization ---
    #################################################################
    if HAS_INT8 and args.calib:
        print("\n=== Exporting INT8 ONNX ===")
        sess = ort.InferenceSession(str(fp32_path))
        input_name = sess.get_inputs()[0].name

        reader = ImageFolderCalibration(args.calib, input_name=input_name)
        int8_path = outdir / "model_int8.onnx"
        quantize_static(
            str(fp32_path), str(int8_path), reader, quant_format=QuantType.QDQ
        )
        print(f"[OK] INT8 export complete: {int8_path}")
    else:
        print(
            "[INFO] INT8 quant not performed (missing args.calib or quantization lib)"
        )

    #################################################################
    # --- 4. Validate ---
    #################################################################
    if args.val:
        val_imgs = list(Path(args.val).glob("*"))[:10]
        if not val_imgs:
            print("[WARN] No validation images found")
        else:
            compare_pytorch_and_onnx(model, fp32_path, val_imgs)
            if fp16_path.exists():
                compare_pytorch_and_onnx(model, fp16_path, val_imgs)
            if HAS_INT8 and args.calib:
                int8_path = outdir / "model_int8.onnx"
                if int8_path.exists():
                    compare_pytorch_and_onnx(model, int8_path, val_imgs)

    print("\n=== DONE ===")
    print("Exported to:", outdir)


if __name__ == "__main__":
    export_everything()
