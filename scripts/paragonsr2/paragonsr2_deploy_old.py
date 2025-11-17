#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import ModelProto
from onnxconverter_common.float16 import convert_float_to_float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxslim import slim
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from traiNNer.archs.paragonsr2_arch import ParagonSR2
from traiNNer.utils.registry import ARCH_REGISTRY

LOGGER = logging.getLogger("paragonsr2.deploy")
SUPPORTED_VARIANTS = sorted(
    name for name in ARCH_REGISTRY.keys() if name.startswith("paragonsr2")
)
DEFAULT_DUMMY_SIZE = 32
DEFAULT_OPSET = 17
DEFAULT_CAL_LIMIT = 128


def ensure_model_proto(candidate: Any, context: str) -> ModelProto:
    if isinstance(candidate, ModelProto):
        return candidate
    if isinstance(candidate, tuple):
        for item in candidate:
            if isinstance(item, ModelProto):
                return item
    raise TypeError(f"{context} did not return an ONNX ModelProto.")


class ImageFolderCalibrationDataReader(CalibrationDataReader):
    """ONNX Runtime calibration reader that yields RGB tensors from an image folder."""

    def __init__(
        self,
        input_name: str,
        image_paths: Sequence[Path],
        limit: int,
        logger: logging.Logger,
    ) -> None:
        self.input_name = input_name
        self.logger = logger
        self._data: list[dict[str, np.ndarray]] = []
        for path in list(image_paths)[:limit]:
            try:
                arr = self._load_image(path)
            except Exception as exc:  # pragma: no cover - I/O errors
                self.logger.warning("Skipping calibration image %s: %s", path, exc)
                continue
            self._data.append({self.input_name: arr})
        self.sample_count = len(self._data)
        self._iterator = iter(self._data)

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        arr = np.expand_dims(arr, axis=0)  # NCHW
        return arr

    def get_next(self) -> dict[str, np.ndarray]:  # type: ignore[override]
        try:
            return next(self._iterator)
        except StopIteration:
            return {}

    def rewind(self) -> None:
        self._iterator = iter(self._data)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if not SUPPORTED_VARIANTS:
        raise RuntimeError("No ParagonSR2 variants registered in ARCH_REGISTRY.")

    parser = argparse.ArgumentParser(
        description="Fuse ParagonSR2 checkpoints and export optimized ONNX bundles."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the unfused ParagonSR2 checkpoint (.safetensors).",
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=SUPPORTED_VARIANTS,
        help="Registered ParagonSR2 variant used during training (e.g., paragonsr2_tiny).",
    )
    parser.add_argument(
        "--scale",
        required=True,
        type=int,
        help="Training scale factor (e.g., 2 for 2×).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where fused checkpoints and ONNX exports will be written.",
    )
    parser.add_argument(
        "--calibration-dir",
        help="Directory of RGB images used for INT8 calibration. If omitted, INT8 export is skipped.",
    )
    parser.add_argument(
        "--calibration-limit",
        type=int,
        default=DEFAULT_CAL_LIMIT,
        help=f"Maximum number of images to use for INT8 calibration (default: {DEFAULT_CAL_LIMIT}).",
    )
    parser.add_argument(
        "--dummy-size",
        type=int,
        default=DEFAULT_DUMMY_SIZE,
        help=f"Spatial size for the dummy input tensor during ONNX export (default: {DEFAULT_DUMMY_SIZE}).",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"Opset version to use for ONNX export (default: {DEFAULT_OPSET}).",
    )
    parser.add_argument(
        "--disable-slim",
        action="store_true",
        help="Disable ONNXSlim graph optimizations.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX vs PyTorch numerical verification.",
    )
    return parser.parse_args(argv)


def collect_image_paths(folder: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)


def load_variant(variant: str, scale: int, logger: logging.Logger) -> ParagonSR2:
    key = variant.lower()
    if key not in ARCH_REGISTRY:
        raise ValueError(
            f"Variant '{variant}' is not registered. Available options: {', '.join(SUPPORTED_VARIANTS)}"
        )
    constructor = ARCH_REGISTRY.get(key)
    logger.info("Instantiating %s (scale=%d)...", variant, scale)
    model = constructor(scale=scale)
    if not isinstance(model, ParagonSR2):
        raise TypeError(f"Registered variant '{variant}' did not return a ParagonSR2.")
    return model


def extract_state_dict(raw_state: dict[str, Any]) -> dict[str, Tensor]:
    candidate_keys = ("params_ema", "params", "state_dict", "model", "net")
    weights: Any = None
    for key in candidate_keys:
        if key in raw_state:
            weights = raw_state[key]
            break
    if weights is None:
        weights = raw_state

    if not isinstance(weights, dict):
        raise RuntimeError("Checkpoint does not contain a valid state dictionary.")

    sample_key = next(iter(weights.keys()))
    if sample_key.startswith("module."):
        weights = {k.replace("module.", "", 1): v for k, v in weights.items()}

    return cast(dict[str, Tensor], weights)


def load_checkpoint_into(
    model: nn.Module, checkpoint: Path, logger: logging.Logger
) -> None:
    logger.info("Loading checkpoint: %s", checkpoint)
    raw_state = load_file(str(checkpoint))
    state_dict = extract_state_dict(raw_state)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded successfully.")


def create_dummy_input(
    scale: int, device: torch.device, size: int = DEFAULT_DUMMY_SIZE
) -> Tensor:
    if size <= 0:
        raise ValueError("Dummy input size must be positive.")
    return torch.randn(1, 3, size, size, device=device)


def verify_onnx(
    model: nn.Module,
    dummy_input: Tensor,
    onnx_path: Path,
    logger: logging.Logger,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> None:
    logger.info("Verifying ONNX output against PyTorch for %s", onnx_path.name)
    model.eval()
    with torch.no_grad():
        torch_output = model(dummy_input).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = np.asarray(session.run(None, ort_inputs)[0])
    np.testing.assert_allclose(torch_output, ort_output, rtol=rtol, atol=atol)
    logger.info("Verification succeeded.")


def export_onnx_models(
    model: nn.Module,
    dummy_input: Tensor,
    output_dir: Path,
    name_stem: str,
    opset: int,
    optimize: bool,
    verify: bool,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    logger.info("Exporting FP32 ONNX (opset=%d)...", opset)
    raw_path = output_dir / f"{name_stem}_fp32_raw.onnx"
    fp32_path = output_dir / f"{name_stem}_fp32_op{opset}_onnxslim.onnx"
    fp16_path = output_dir / f"{name_stem}_fp16_op{opset}_onnxslim.onnx"

    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        model,
        (dummy_input,),
        raw_path,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    onnx_model = ensure_model_proto(onnx.load(str(raw_path)), "Initial ONNX export")
    if optimize:
        onnx_model = ensure_model_proto(
            slim(onnx_model), "ONNXSlim optimization (FP32)"
        )
    save_model(onnx_model, str(fp32_path))
    if raw_path.exists():
        raw_path.unlink()

    if verify:
        verify_onnx(model, dummy_input, fp32_path, logger)

    logger.info("Creating FP16 ONNX...")
    fp16_model = convert_float_to_float16(onnx_model)
    save_model(fp16_model, str(fp16_path))
    return fp32_path, fp16_path


def quantize_with_calibration(
    fp32_path: Path,
    output_dir: Path,
    name_stem: str,
    opset: int,
    calibration_dir: Path,
    limit: int,
    logger: logging.Logger,
) -> Path | None:
    if not calibration_dir.is_dir():
        logger.error("Calibration directory %s does not exist.", calibration_dir)
        return None

    image_paths = collect_image_paths(calibration_dir)
    if not image_paths:
        logger.warning(
            "No calibration images found in %s; skipping INT8 export.", calibration_dir
        )
        return None

    logger.info(
        "Preparing INT8 calibration with up to %d images from %s",
        limit,
        calibration_dir,
    )
    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reader = ImageFolderCalibrationDataReader(input_name, image_paths, limit, logger)

    if reader.sample_count == 0:
        logger.warning("All calibration images failed to load; skipping INT8 export.")
        return None

    int8_tmp = output_dir / f"{name_stem}_int8_tmp.onnx"
    int8_path = output_dir / f"{name_stem}_int8_op{opset}_onnxslim.onnx"

    logger.info("Running ONNX Runtime static quantization...")
    try:
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(int8_tmp),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.Entropy,
        )
    except Exception as exc:  # pragma: no cover - quantization failure
        logger.exception("INT8 quantization failed: %s", exc)
        if int8_tmp.exists():
            int8_tmp.unlink()
        return None

    int8_model = ensure_model_proto(
        onnx.load(str(int8_tmp)), "INT8 quantized model load"
    )
    int8_model = ensure_model_proto(slim(int8_model), "ONNXSlim optimization (INT8)")
    save_model(int8_model, str(int8_path))
    if int8_tmp.exists():
        int8_tmp.unlink()
    logger.info("Saved INT8 ONNX model to %s", int8_path)
    return int8_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        LOGGER.error("Checkpoint file %s does not exist.", checkpoint)
        return 1

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_variant(args.variant, args.scale, LOGGER).to(device)
    load_checkpoint_into(model, checkpoint, LOGGER)
    model.eval()

    LOGGER.info("Creating fused copy for deployment...")
    fused_model = copy.deepcopy(model).to("cpu")
    fused_model.eval()
    fused_model = fused_model.fuse_for_release()

    fused_state = {k: v.detach().cpu() for k, v in fused_model.state_dict().items()}
    fused_path = output_dir / f"{checkpoint.stem}_fused.safetensors"
    save_file(fused_state, str(fused_path))
    LOGGER.info("Saved fused checkpoint to %s", fused_path)

    dummy_input = create_dummy_input(args.scale, torch.device("cpu"), args.dummy_size)
    with torch.no_grad():
        fused_model(dummy_input)

    fp32_path, fp16_path = export_onnx_models(
        fused_model,
        dummy_input,
        output_dir,
        checkpoint.stem,
        args.onnx_opset,
        optimize=not args.disable_slim,
        verify=not args.skip_verify,
        logger=LOGGER,
    )

    int8_path = None
    if args.calibration_dir:
        int8_path = quantize_with_calibration(
            fp32_path,
            output_dir,
            checkpoint.stem,
            args.onnx_opset,
            Path(args.calibration_dir).expanduser().resolve(),
            args.calibration_limit,
            LOGGER,
        )

    LOGGER.info("Deployment artifacts:")
    LOGGER.info("  Fused checkpoint: %s", fused_path)
    LOGGER.info("  ONNX FP32:        %s", fp32_path)
    LOGGER.info("  ONNX FP16:        %s", fp16_path)
    if int8_path:
        LOGGER.info("  ONNX INT8:        %s", int8_path)
    else:
        LOGGER.info("  ONNX INT8:        skipped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
