#!/usr/bin/env python3
"""
ParagonSR2 Benchmark Tool
=========================

Benchmarks ParagonSR2 models on real images with both PyTorch and TensorRT backends.
Outputs results in markdown table format for easy README integration.

Features:
- PyTorch FP32, FP16 (AMP), and Compiled benchmarks
- TensorRT FP16 engine benchmarks
- Per-image timing to show dynamic shape handling
- System info collection (GPU, driver, VRAM)
- Markdown output for README copy-paste

Usage:
    # PyTorch only
    python benchmark_release.py \\
        --input /path/to/Urban100_x2 \\
        --scale 2 \\
        --pt_model paragonsr2/2xParagonSR2_Photo_fidelity.safetensors \\
        --arch paragonsr2_photo

    # PyTorch + TensorRT comparison
    python benchmark_release.py \\
        --input /path/to/Urban100_x2 \\
        --scale 2 \\
        --pt_model paragonsr2/2xParagonSR2_Photo_fidelity.safetensors \\
        --arch paragonsr2_photo \\
        --trt_engine paragonsr2_release/paragonsr2_photo_fp16.trt

Author: Philip Hofmann
License: MIT
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=""):
        return iterable


# Try importing TensorRT
try:
    import tensorrt as trt

    HAS_TRT = True
except ImportError:
    HAS_TRT = False

# Try importing FlexAttention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    HAS_FLEX = True
except (ImportError, AttributeError):
    HAS_FLEX = False


# =============================================================================
# SYSTEM INFO
# =============================================================================


def get_system_info() -> dict:
    """Collect system information for benchmark context."""
    info = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        if torch.cuda.is_available()
        else "N/A",
        "cuda_version": torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

    # Try to get driver version
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["driver"] = result.stdout.strip()
    except Exception:
        info["driver"] = "N/A"

    return info


def print_system_info(info: dict) -> None:
    """Print system info in a nice format."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"  GPU:          {info['gpu']}")
    print(f"  VRAM:         {info['gpu_vram_total']}")
    print(f"  Driver:       {info['driver']}")
    print(f"  CUDA:         {info['cuda_version']}")
    print(f"  PyTorch:      {info['pytorch_version']}")
    print(f"  OS:           {info['os']}")
    print("=" * 60)


# =============================================================================
# PYTORCH RUNNER
# =============================================================================


class PyTorchRunner:
    """
    PyTorch model runner for benchmarking.

    Supports FP32, FP16 (AMP), and torch.compile modes.
    """

    def __init__(
        self,
        model_path: str,
        arch: str,
        scale: int,
        device: str = "cuda",
        use_content_aware: bool | None = None,
        upsampler_alpha: float | None = None,
        detail_gain: float | None = None,
        attention_mode: str | None = None,
        export_safe: bool | None = None,
        window_size: int | None = None,
    ) -> None:
        self.device = device
        self.scale = scale
        self.is_compiled = False
        print(f"[PyTorch] Loading {arch} from {model_path}...")

        # Add repo root to path for traiNNer imports
        repo_root = Path(__file__).parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from traiNNer.archs import paragonsr2_arch

        arch_map = {
            "paragonsr2_realtime": paragonsr2_arch.paragonsr2_realtime,
            "paragonsr2_stream": paragonsr2_arch.paragonsr2_stream,
            "paragonsr2_photo": paragonsr2_arch.paragonsr2_photo,
        }
        if arch not in arch_map:
            raise ValueError(f"Unknown arch: {arch}")

        # Build kwargs with overrides
        arch_kwargs = {"scale": scale}
        if use_content_aware is not None:
            arch_kwargs["use_content_aware"] = use_content_aware
            print(f"      Override: use_content_aware={use_content_aware}")
        if upsampler_alpha is not None:
            arch_kwargs["upsampler_alpha"] = upsampler_alpha
            print(f"      Override: upsampler_alpha={upsampler_alpha}")
        if detail_gain is not None:
            arch_kwargs["detail_gain"] = detail_gain
            print(f"      Override: detail_gain={detail_gain}")
        if attention_mode is not None:
            arch_kwargs["attention_mode"] = attention_mode
            print(f"      Override: attention_mode={attention_mode}")
        if export_safe is not None:
            arch_kwargs["export_safe"] = export_safe
            print(f"      Override: export_safe={export_safe}")
        if window_size is not None:
            arch_kwargs["window_size"] = window_size
            print(f"      Override: window_size={window_size}")

        self.model = arch_map[arch](**arch_kwargs)

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

        # Legacy key mapping (backwards compatibility)
        new_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith("module.") else k
            new_k = new_k.replace("base_upsampler.resample_conv.", "base.blur.")
            new_k = new_k.replace("base_upsampler.sharpen.", "base.sharp.")
            if new_k.startswith("base."):
                new_k = new_k.replace(".conv_h.", ".h.")
                new_k = new_k.replace(".conv_v.", ".v.")
            if ".blocks." in new_k and ".net." in new_k:
                new_k = new_k.replace(".net.0.", ".conv1.")
                new_k = new_k.replace(".net.2.", ".dw.")
                new_k = new_k.replace(".net.4.", ".conv2.")
            new_k = new_k.replace("global_detail_gain", "detail_gain")
            new_k = new_k.replace("detail_upsampler.up_conv.", "up.0.")
            new_k = new_k.replace("conv_fuse.", "conv_mid.")
            new_dict[new_k] = v

        self.model.load_state_dict(new_dict, strict=False)
        self.model.to(device).eval()

        # Count parameters
        params = sum(p.numel() for p in self.model.parameters())
        print(f"      Parameters: {params:,}")

    def infer(self, input_tensor: torch.Tensor, use_amp: bool = False) -> torch.Tensor:
        """Run inference on input tensor."""
        if self.is_compiled:
            input_tensor = input_tensor.contiguous()

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(input_tensor)
            return self.model(input_tensor)

    def compile_model(self) -> None:
        """Compile model with torch.compile for faster inference."""
        # Use 'default' mode which is the most robust for dynamic shapes.
        # 'reduce-overhead' and 'max-autotune' enable CUDA graphs which
        # crash on varying stride/shape patterns in differing images.
        print("      Compiling model (mode='default', dynamic=True)...")
        self.model = torch.compile(self.model, mode="default", dynamic=True)
        self.is_compiled = True


# =============================================================================
# TENSORRT RUNNER
# =============================================================================


class TRTRunner:
    """
    TensorRT engine runner for benchmarking.

    Supports dynamic shapes and optional feature_map output for video mode.
    """

    def __init__(self, engine_path: str, is_video: bool = False) -> None:
        if not HAS_TRT:
            raise RuntimeError("TensorRT not installed")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.is_video = is_video

        print(f"[TRT] Loading Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def run(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run TensorRT inference."""
        input_ptr = input_tensor.data_ptr()

        # Set input shape for dynamic engines
        self.context.set_input_shape("input", input_tensor.shape)

        # Get output shape and dtype
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


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================


def benchmark_pytorch(
    runner: PyTorchRunner,
    images: list[Path],
    scale: int,
    use_amp: bool = False,
    device: str = "cuda",
    mode_suffix: str = "",
) -> dict:
    """Benchmark PyTorch model on a list of images."""
    mode = ("PyTorch FP16" if use_amp else "PyTorch FP32") + mode_suffix
    print(f"\n--- Benchmarking {mode} ---")

    latencies = []
    image_sizes = []

    torch.cuda.reset_peak_memory_stats()

    # Warmup with a standard size (skip for compiled to prevent static shape baking)
    if not runner.is_compiled:
        warmup_size = 64
        dummy = torch.zeros(1, 3, warmup_size, warmup_size, device=device)
        dummy = dummy.to(memory_format=torch.channels_last)
        for _ in range(3):
            runner.infer(dummy, use_amp=use_amp)
        torch.cuda.synchronize()

    for img_path in tqdm(images, desc=mode):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        image_sizes.append((h, w))

        img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.div_(255.0)

        # For compiled models, ensure contiguous memory for safer dynamic handling
        # For non-compiled, channels_last is usually faster
        if runner.is_compiled:
            img_t = img_t.contiguous()
            # Mark dynamic dimensions to prevent specialization crashes
            if hasattr(torch, "_dynamo"):
                torch._dynamo.mark_dynamic(img_t, 2)
                torch._dynamo.mark_dynamic(img_t, 3)

            # Dynamic shape warmup (crucial for Inductor stability on varying shapes)
            if len(latencies) == 0:
                _ = runner.infer(img_t, use_amp=use_amp)
        else:
            img_t = img_t.to(memory_format=torch.channels_last)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = runner.infer(img_t, use_amp=use_amp)
        end_event.record()
        torch.cuda.synchronize()

        latencies.append(start_event.elapsed_time(end_event))

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency

    print(f"  Avg Latency:  {avg_latency:.2f} ms")
    print(f"  Throughput:   {fps:.2f} FPS")
    print(f"  Peak VRAM:    {peak_vram:.3f} GB")

    return {
        "mode": mode,
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "peak_vram_gb": peak_vram,
        "image_count": len(latencies),
        "image_sizes": image_sizes,
    }


def benchmark_tensorrt(
    engine_path: str,
    images: list[Path],
    scale: int,
    is_video: bool = False,
    device: str = "cuda",
) -> dict:
    """Benchmark TensorRT engine on a list of images."""
    mode = f"TensorRT FP16{' (Video)' if is_video else ''}"
    print(f"\n--- Benchmarking {mode} ---")

    try:
        runner = TRTRunner(engine_path, is_video=is_video)
    except Exception as e:
        print(f"Error initializing TRTRunner: {e}")
        return {}

    latencies = []
    image_sizes = []

    torch.cuda.reset_peak_memory_stats()

    # Warmup
    dummy = torch.zeros(1, 3, 64, 64, device=device, dtype=torch.float32)
    for _ in range(3):
        runner.run(dummy)
    torch.cuda.synchronize()

    for img_path in tqdm(images, desc=mode):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        image_sizes.append((h, w))

        img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.div_(255.0)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = runner.run(img_t)
        end_event.record()
        torch.cuda.synchronize()

        latencies.append(start_event.elapsed_time(end_event))

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency

    print(f"  Avg Latency:  {avg_latency:.2f} ms")
    print(f"  Throughput:   {fps:.2f} FPS")
    print(f"  Peak VRAM:    {peak_vram:.3f} GB")

    return {
        "mode": mode,
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "peak_vram_gb": peak_vram,
        "image_count": len(latencies),
        "image_sizes": image_sizes,
    }


def print_markdown_table(results: list[dict], arch: str, system_info: dict) -> None:
    """Print results as markdown table for README."""
    print("\n" + "=" * 60)
    print("MARKDOWN OUTPUT (Copy to README)")
    print("=" * 60)

    print(f"\n### Benchmark: `{arch}` (2x)")
    print(f"\n**Hardware:** {system_info['gpu']} ({system_info['gpu_vram_total']})")
    print(f"**Dataset:** Urban100 ({results[0]['image_count']} images, varied sizes)")
    print()
    print("| Backend | Avg Latency | FPS | Peak VRAM |")
    print("|---------|-------------|-----|-----------|")

    for r in results:
        print(
            f"| {r['mode']} | {r['avg_latency_ms']:.1f} ms | {r['fps']:.1f} | {r['peak_vram_gb']:.2f} GB |"
        )

    print()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="ParagonSR2 Benchmark Tool")
    parser.add_argument("--input", required=True, help="Folder of test images")
    parser.add_argument("--scale", type=int, default=2)

    # PyTorch Args
    parser.add_argument("--pt_model", help="Path to .safetensors model")
    parser.add_argument(
        "--arch", help="Model architecture name (e.g. paragonsr2_photo)"
    )

    # TRT Args
    parser.add_argument("--trt_engine", help="Path to .trt engine")

    # Architecture overrides
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
    parser.add_argument(
        "--benchmark_attention_modes",
        action="store_true",
        help="Benchmark No-Attn, SDPA, and Flex variants (Photo arch only)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable feature_tap benchmarking (video temporal stability mode)",
    )

    args = parser.parse_args()

    # Collect images
    images = sorted(Path(args.input).glob("*"))
    images = [p for p in images if p.suffix.lower() in [".jpg", ".png", ".webp"]]

    if not images:
        print("No images found.")
        return

    # System info
    system_info = get_system_info()
    print_system_info(system_info)

    print(f"\nBenchmarking on {len(images)} images (Scale: {args.scale}x)")

    results = []

    # PyTorch Benchmarks
    if args.pt_model and args.arch:
        # Helper to run a config
        def run_config(attn_mode=None, exp_safe=None, label_suffix="", compile=False):
            runner = PyTorchRunner(
                args.pt_model,
                args.arch,
                args.scale,
                use_content_aware=args.use_content_aware,
                upsampler_alpha=args.upsampler_alpha,
                detail_gain=args.detail_gain,
                attention_mode=attn_mode
                if attn_mode is not None
                else args.attention_mode,
                export_safe=exp_safe if exp_safe is not None else args.export_safe,
                window_size=args.window_size,
            )
            if compile:
                runner.compile_model()
                label_suffix += " (Compiled)"

            res = benchmark_pytorch(
                runner, images, args.scale, use_amp=True, mode_suffix=label_suffix
            )
            del runner
            torch.cuda.empty_cache()
            return res

        if args.benchmark_attention_modes and "photo" in args.arch:
            print("\n>>> Running Attention Mode Comparison Suite <<<")

            # 1. No Attention (Export Safe)
            print("1. Testing: No Attention (Export Safe aka CNN-only)")
            results.append(
                run_config(
                    attn_mode="sdpa",
                    exp_safe=True,
                    label_suffix=" (No Attn)",
                    compile=False,
                )
            )
            results.append(
                run_config(
                    attn_mode="sdpa",
                    exp_safe=True,
                    label_suffix=" (No Attn)",
                    compile=True,
                )
            )

            # 2. SDPA
            print("2. Testing: SDPA (Standard)")
            results.append(
                run_config(
                    attn_mode="sdpa",
                    exp_safe=False,
                    label_suffix=" (SDPA)",
                    compile=False,
                )
            )
            results.append(
                run_config(
                    attn_mode="sdpa",
                    exp_safe=False,
                    label_suffix=" (SDPA)",
                    compile=True,
                )
            )

            # 3. FlexAttention
            if HAS_FLEX:
                print("3. Testing: FlexAttention")
                # Eager (slow, for reference)
                results.append(
                    run_config(
                        attn_mode="flex",
                        exp_safe=False,
                        label_suffix=" (Flex)",
                        compile=False,
                    )
                )
                # Compiled (fast)
                results.append(
                    run_config(
                        attn_mode="flex",
                        exp_safe=False,
                        label_suffix=" (Flex)",
                        compile=True,
                    )
                )
            else:
                print("3. FlexAttention skipped (not available in this PyTorch build).")

        else:
            # Standard single run logic
            pt_runner = PyTorchRunner(
                args.pt_model,
                args.arch,
                args.scale,
                use_content_aware=args.use_content_aware,
                upsampler_alpha=args.upsampler_alpha,
                detail_gain=args.detail_gain,
                attention_mode=args.attention_mode,
                export_safe=args.export_safe,
                window_size=args.window_size,
            )

            # FP32 benchmark
            results.append(
                benchmark_pytorch(pt_runner, images, args.scale, use_amp=False)
            )
            torch.cuda.empty_cache()

            # FP16 (AMP) benchmark
            results.append(
                benchmark_pytorch(pt_runner, images, args.scale, use_amp=True)
            )
            torch.cuda.empty_cache()

            # Compiled Benchmark
            pt_runner.compile_model()
            results.append(
                benchmark_pytorch(
                    pt_runner,
                    images,
                    args.scale,
                    use_amp=True,
                    mode_suffix=" (Compiled)",
                )
            )
            torch.cuda.empty_cache()
            del pt_runner
            torch.cuda.empty_cache()

    # TensorRT Benchmark
    if args.trt_engine:
        if not HAS_TRT:
            print("TensorRT not installed, skipping TRT benchmark.")
        else:
            results.append(
                benchmark_tensorrt(
                    args.trt_engine, images, args.scale, is_video=args.video
                )
            )
            torch.cuda.empty_cache()

    # Print markdown table
    if results:
        print_markdown_table(results, args.arch or "unknown", system_info)


if __name__ == "__main__":
    main()
