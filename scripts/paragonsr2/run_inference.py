#!/usr/bin/env python3
"""
ParagonSR2 Universal Inference Script
=====================================

"Smart" inference tool that automatically selects the best backend and optimal
settings for Image and Video upscaling.

Features:
    - **Auto-Backend**: Tries TensorRT -> Compiled PyTorch -> Standard PyTorch.
    - **Smart Input**: Handles Single Image, Single Video, or Folder (mixed content).
    - **Video Stabilization**: Automatically enables Feature-Tap Temporal Smoothing
      for videos. Uses prev_feat injection for true temporal consistency.
    - **Scene Detection**: Prevents ghosting on scene cuts by resetting state.
    - **Resumable**: Skips already processed files.

Usage:
    python run_inference.py \\
        --input my_video.mp4 \\
        --model models/paragonsr2_photo.safetensors \\
        --arch paragonsr2_photo \\
        --scale 2

    python run_inference.py \\
        --input /folder/of/images \\
        --model models/paragonsr2_photo.safetensors \\
        --arch paragonsr2_photo \\
        --output /output/folder

        python scripts/paragonsr2/run_inference.py     --input /home/phips/Documents/dataset/Urban100/x2     --model ./paragonsr2/2xParagonSR2_Photo_fidelity.safetensors     --arch paragonsr2_photo     --output /home/phips/Documents/dataset/Urban100/upscaled     --fp16     --upsampler_alpha 0.0 --scale 2


Author: Philip Hofmann
License: MIT
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import tensorrt as trt

    HAS_TRT = True
except ImportError:
    HAS_TRT = False

# Import Architecture
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import paragonsr2_arch
except ImportError:
    try:
        repo_root = Path(__file__).parents[2]
        sys.path.insert(0, str(repo_root))
        from traiNNer.archs import paragonsr2_arch
    except ImportError:
        print("Error: Could not find paragonsr2_arch.py")
        sys.exit(1)


# =============================================================================
# BACKEND RUNNERS
# =============================================================================


class TRTRunner:
    """TensorRT engine runner with optional feature_map output for video mode."""

    def __init__(self, engine_path: str, is_video: bool = False) -> None:
        if not HAS_TRT:
            raise RuntimeError("TensorRT not installed")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.is_video = is_video

        print(f"      [Backend] Loading TensorRT Engine: {Path(engine_path).name}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def run(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Run TensorRT inference."""
        input_ptr = input_tensor.data_ptr()
        self.context.set_input_shape("input", input_tensor.shape)

        out_shape = self.context.get_tensor_shape("output")
        out_dtype = (
            torch.float16
            if self.engine.get_tensor_dtype("output") == trt.float16
            else torch.float32
        )
        output_tensor = torch.empty(
            tuple(out_shape), dtype=out_dtype, device=input_tensor.device
        )

        bindings = [input_ptr, output_tensor.data_ptr()]

        feat_tensor = None
        if self.is_video and self.engine.num_io_tensors > 2:
            feat_shape = self.context.get_tensor_shape("feature_map")
            feat_dtype = (
                torch.float16
                if self.engine.get_tensor_dtype("feature_map") == trt.float16
                else torch.float32
            )
            feat_tensor = torch.empty(
                tuple(feat_shape), dtype=feat_dtype, device=input_tensor.device
            )
            bindings.append(feat_tensor.data_ptr())

        self.context.execute_v2(bindings)

        if self.is_video:
            return output_tensor, feat_tensor
        return output_tensor


class PyTorchRunner:
    """PyTorch model runner with optional torch.compile and FP16 support."""

    def __init__(
        self,
        model_path: str,
        arch: str,
        scale: int,
        device: str = "cuda",
        compile: bool = True,
        fp16: bool = True,
        upsampler_alpha: float | None = None,
        export_safe: bool = False,
    ) -> None:
        self.device = device
        self.warmup_done = False
        self.fp16 = fp16

        self.compiled = False
        self.warmup_done = False

        print(f"      [Backend] Loading PyTorch Model: {Path(model_path).name}")

        arch_map = {
            "paragonsr2_realtime": paragonsr2_arch.paragonsr2_realtime,
            "paragonsr2_stream": paragonsr2_arch.paragonsr2_stream,
            "paragonsr2_photo": paragonsr2_arch.paragonsr2_photo,
            "paragonsr2_pro": paragonsr2_arch.paragonsr2_pro,
        }
        if arch not in arch_map:
            raise ValueError(f"Unknown architecture: {arch}")

        # Build model options
        model_kwargs = {"scale": scale, "export_safe": export_safe}
        if upsampler_alpha is not None:
            model_kwargs["upsampler_alpha"] = upsampler_alpha

        self.model = arch_map[arch](**model_kwargs)

        # Load weights
        if str(model_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]

        # Legacy key mapping for backwards compatibility
        new_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("base_upsampler.resample_conv.", "base.blur.")
            k = k.replace("base_upsampler.sharpen.", "base.sharp.")
            k = k.replace("global_detail_gain", "detail_gain")
            k = k.replace("conv_fuse.", "conv_mid.")
            k = k.replace("detail_upsampler.up_conv.", "up.0.")
            if ".blocks." in k and ".net." in k:
                k = k.replace(".net.0.", ".conv1.")
                k = k.replace(".net.2.", ".dw.")
                k = k.replace(".net.4.", ".conv2.")
            new_dict[k] = v

        self.model.load_state_dict(new_dict, strict=False)
        self.model.to(device).eval()

        if fp16:
            self.model.half()

        if compile:
            print(
                "      [Backend] Compiling model (default mode for dynamic robustnes)..."
            )
            # 'default' is robust for dynamic shapes; 'reduce-overhead' crashes on varying inputs
            self.model = torch.compile(self.model, mode="default", dynamic=True)
            self.compiled = True
            self.warmup_done = False

    def run(
        self,
        input_tensor: torch.Tensor,
        feature_tap: bool = False,
        prev_feat: torch.Tensor | None = None,
        force_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run PyTorch inference with optional feature tap."""
        if self.compiled and not force_fp32:
            input_tensor = input_tensor.contiguous()
            if prev_feat is not None:
                prev_feat = prev_feat.contiguous()

            # Critical: Mark dynamic dimensions to prevent compiler crashing on varying shapes
            if hasattr(torch, "_dynamo"):
                # Mark H (2) and W (3) as dynamic
                torch._dynamo.mark_dynamic(input_tensor, 2)
                torch._dynamo.mark_dynamic(input_tensor, 3)

            # Just-in-time warmup for the first run if needed
            if not self.warmup_done:
                self.warmup_done = True
                # Simple warmup call
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=self.device, dtype=torch.float16),
                ):
                    _ = self.model(
                        input_tensor, feature_tap=feature_tap, prev_feat=prev_feat
                    )

        # Determine precision
        use_fp16 = self.fp16 and not force_fp32

        # If forcing FP32 on a compiled model, we might need to use the original model
        # if the compiled one is hard-coded for something else, but torch.compile usually handles it.
        # However, for maximum safety during fallback, it might be safer to use the original model object
        # if we could access it. But self.model is overwritten.
        # We rely on autocast(enabled=False) to force FP32.

        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    out = self.model(
                        input_tensor, feature_tap=feature_tap, prev_feat=prev_feat
                    )
            else:
                # Force FP32 context
                # If the model is half(), we need to cast input to half? No, if we want FP32, we want float().
                # But if self.model.half() was called, weights are half.
                # executing half weights in float32 autocast?
                # If model is half, we must run in half or cast model back.
                # Re-casting model is expensive.
                # If self.fp16 was set, self.model.half() was called.
                # If we want to fix NaNs, usually we need FP32 weights + FP32 ops.
                # But maybe running FP16 weights in FP32 accumulation helps? (autocast disabled).
                # Wait, if weights are FP16, running without autocast expects inputs to be FP16.
                # And ops will be FP16.

                # If model is converted to half (line 193), we cannot easily run in FP32
                # without converting back to float.
                pass

        # To properly support force_fp32 fallback when model is .half():
        # We must cast the model to float temporarily or keep a float copy.
        # Keeping a copy is memory expensive.
        # Casting back and forth is acceptable for a "Retry" scenario (slow but works).

        if force_fp32 and self.fp16:
            # Cast model to float, run, cast back
            # Note: maximizing stability over speed here
            self.model.float()
            # Input must be float (it is created as float in process_*, only autocast handles the rest)
            # but we need to make sure input_tensor is float.
            input_tensor = input_tensor.float()
            if prev_feat is not None:
                prev_feat = prev_feat.float()

            out = self.model(input_tensor, feature_tap=feature_tap, prev_feat=prev_feat)

            # Restore half
            self.model.half()
        else:
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device,
                    dtype=torch.float16 if use_fp16 else torch.float32,
                ):
                    out = self.model(
                        input_tensor, feature_tap=feature_tap, prev_feat=prev_feat
                    )

        if feature_tap:
            return out[0], out[1]
        return out


# =============================================================================
# INFERENCE ORCHESTRATOR
# =============================================================================


class InferenceOrchestrator:
    """
    Main inference coordinator.

    Handles backend selection, image/video processing, and temporal state management.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runner = None
        self.mode = None  # 'trt', 'pt_compiled', 'pt'

        self._init_backend()

    def _init_backend(self) -> None:
        """Intelligently select and initialize the best available backend."""
        model_path = Path(self.args.model)

        # 1. Try TensorRT first (fastest)
        trt_path = model_path.with_suffix(".trt")
        if model_path.suffix == ".trt":
            trt_path = model_path

        if HAS_TRT and trt_path.exists():
            try:
                self.runner = TRTRunner(str(trt_path), is_video=True)
                self.mode = "trt"
                print("[Init] Selected Backend: TensorRT (Fastest)")
                return
            except Exception as e:
                print(f"[Init] Failed to load TensorRT engine: {e}")

        # 2. Try PyTorch (Compiled or Standard)
        if model_path.suffix in [".pth", ".pt", ".safetensors"]:
            try:
                use_compile = not self.args.disable_compile
                if self.device == "cpu":
                    use_compile = False

                self.runner = PyTorchRunner(
                    str(model_path),
                    self.args.arch,
                    self.args.scale,
                    device=self.device,
                    compile=use_compile,
                    fp16=self.args.fp16,
                    upsampler_alpha=self.args.upsampler_alpha,
                    export_safe=self.args.export_safe,
                )
                self.mode = "pt_compiled" if use_compile else "pt"
                print(
                    f"[Init] Selected Backend: {'Compiled ' if use_compile else ''}PyTorch"
                )
                return
            except Exception as e:
                print(f"[Init] Failed to load PyTorch model: {e}")

        raise RuntimeError("No suitable backend could be initialized.")

    def process_image(self, img: np.ndarray) -> np.ndarray:
        """Process a single image (stateless)."""
        img_t = (
            torch.from_numpy(img).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
            / 255.0
        )
        if self.mode == "pt_compiled":
            img_t = img_t.contiguous()

        # RUN INFERENCE
        def _run(force_fp32=False):
            if self.mode == "trt":
                # TRT cannot force FP32 easily
                out = self.runner.run(img_t)
            else:
                out = self.runner.run(img_t, feature_tap=False, force_fp32=force_fp32)

            if isinstance(out, tuple):
                out = out[0]
            return out

        out_t = _run(force_fp32=False)
        out_img = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # NaN/Inf Detection & Retry
        if np.isnan(out_img).any() or np.isinf(out_img).any():
            if self.mode != "trt":
                print(
                    "    [Warning] Numerical instability detected. Retrying with FP32..."
                )
                try:
                    out_t = _run(force_fp32=True)
                    out_img = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    # Check again
                    if np.isnan(out_img).any() or np.isinf(out_img).any():
                        print(
                            "    [Error] FP32 retry failed. Output still contains NaNs."
                        )
                    else:
                        print("    [Success] FP32 retry successful.")
                except Exception as e:
                    print(f"    [Error] FP32 retry failed with exception: {e}")
            else:
                print(
                    "    [Warning] Numerical instability detected in TRT! Cannot retry in FP32."
                )

        out_img = np.clip(out_img, 0, 1) * 255.0
        return out_img.round().astype(np.uint8)

    def process_video(self, input_path: Path, output_path: Path) -> None:
        """
        Process video with Temporal Feature-Tap stabilization.

        Uses prev_feat injection to blend features across frames for smoother output.
        Scene change detection resets the temporal state to prevent ghosting.
        """
        print(f"  [Video] Processing: {input_path.name}")

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale = self.args.scale
        out_width = width * scale
        out_height = height * scale

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_width, out_height),
        )

        # Temporal state
        prev_feat = None
        prev_luma = None
        scene_threshold = 30.0  # Luma difference threshold for scene cut

        pbar = tqdm(total=total_frames, unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Scene cut detection (reset state on scene change)
            current_luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
            if prev_luma is not None:
                if abs(current_luma - prev_luma) > scene_threshold:
                    prev_feat = None  # Reset temporal state
            prev_luma = current_luma

            # Preprocess
            img_t = (
                torch.from_numpy(frame_rgb)
                .to(self.device)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )
            if self.mode == "pt_compiled":
                img_t = img_t.contiguous()

            # Inner run function to handle potential retry
            def _run_video(p_feat, force_fp32=False):
                if self.mode == "trt":
                    res = self.runner.run(img_t)
                    if isinstance(res, tuple):
                        return res[0], res[1]
                    return res, None
                else:
                    res = self.runner.run(
                        img_t, feature_tap=True, prev_feat=p_feat, force_fp32=force_fp32
                    )
                    if isinstance(res, tuple):
                        return res[0], res[1]
                    return res, None

            # First attempt
            out_t, new_feat = _run_video(prev_feat, force_fp32=False)

            # Check for NaNs
            # Optimization: Check GPU tensor directly or CPU numpy?
            # Checking GPU tensor for nan is fast.
            if torch.isnan(out_t).any() or torch.isinf(out_t).any():
                if self.mode != "trt":
                    # Retry in FP32
                    # Note: We must be careful if prev_feat was FP16. _run_video/run handles float() cast internally.
                    out_t, new_feat = _run_video(prev_feat, force_fp32=True)
                else:
                    pass  # Can't fix TRT

            # Update temporal state for next frame
            # Detach to separate graph
            if new_feat is not None:
                prev_feat = new_feat.detach()

            # Postprocess
            out_img = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

            out_img = np.clip(out_img, 0, 1) * 255.0
            out_bgr = cv2.cvtColor(out_img.round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)
            pbar.update(1)

        cap.release()
        writer.release()
        pbar.close()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParagonSR2 Smart Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input image, video, or folder")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model file (.safetensors, .pth, or .trt)",
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Model architecture (paragonsr2_realtime/stream/photo)",
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="Upscale factor (default: 4)"
    )
    parser.add_argument(
        "--output", default="output", help="Output folder (default: output)"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 inference (disables FP16). More stable but slower.",
    )
    parser.add_argument(
        "--upsampler_alpha",
        type=float,
        default=None,
        help="Sharpening strength (0.0-1.0). Default: Arch specific (Photo=0.4, Stream=0.0)",
    )
    parser.add_argument(
        "--export_safe",
        action="store_true",
        help="Disable attention for compatibility (may reduce quality)",
    )
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        help="Disable torch.compile (for debugging)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Re-map fp16 based on fp32 flag
    args.fp16 = not args.fp32

    orchestrator = InferenceOrchestrator(args)

    # Collect files
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            [
                p
                for p in input_path.glob("*")
                if p.suffix.lower()
                in [".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mkv", ".avi", ".mov"]
            ]
        )

    print(f"\nProcessing {len(files)} files...")

    for f in files:
        is_video = f.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov"]
        out_name = f.stem + "_paragonsr2" + (".mp4" if is_video else ".png")
        out_path = output_dir / out_name

        if out_path.exists():
            print(f"Skipping {f.name} (output exists)")
            continue

        if is_video:
            orchestrator.process_video(f, out_path)
        else:
            print(f"  [Image] Processing: {f.name}")
            img = cv2.imread(str(f))
            if img is None:
                print(f"    Warning: Could not read {f.name}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = orchestrator.process_image(img)
            cv2.imwrite(str(out_path), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

    print("\nDone!")


if __name__ == "__main__":
    main()
