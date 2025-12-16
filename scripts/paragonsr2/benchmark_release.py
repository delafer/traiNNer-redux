#!/usr/bin/env python3
"""
ParagonSR2 Benchmark Tool
=========================

Benchmarks ParagonSR2 models on real images with both PyTorch and TensorRT backends.
Outputs results in markdown table format for easy README integration.

Features:
- PyTorch FP32 and FP16 (AMP) benchmarks
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

    # With architecture overrides (for Stream fidelity checkpoint)
    python benchmark_release.py \\
        --input /path/to/Urban100_x2 \\
        --scale 2 \\
        --pt_model paragonsr2/2xParagonSR2_Stream_fidelity.safetensors \\
        --arch paragonsr2_stream \\
        --use_content_aware false
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=""):
        return iterable


import cv2
import numpy as np
import torch

# Try importing TensorRT
try:
    import tensorrt as trt

    HAS_TRT = True
except ImportError:
    HAS_TRT = False


# ---------------------------------------------------------------------
# SYSTEM INFO
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# PYTORCH RUNNER
# ---------------------------------------------------------------------


class PyTorchRunner:
    def __init__(
        self,
        model_path: str,
        arch: str,
        scale: int,
        device: str = "cuda",
        use_content_aware: bool | None = None,
        upsampler_alpha: float | None = None,
        detail_gain: float | None = None,
    ) -> None:
        self.device = device
        self.scale = scale
        print(f"[PyTorch] Loading {arch} from {model_path}...")

        try:
            # Add repo root to path for traiNNer imports
            repo_root = Path(__file__).parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            from traiNNer.archs import paragonsr2_arch

            arch_map = {
                "paragonsr2_realtime": paragonsr2_arch.paragonsr2_realtime,
                "paragonsr2_stream": paragonsr2_arch.paragonsr2_stream,
                "paragonsr2_photo": paragonsr2_arch.paragonsr2_photo,
                "paragonsr2_pro": paragonsr2_arch.paragonsr2_pro,
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

            model = arch_map[arch](**arch_kwargs)

            if str(model_path).endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location="cpu")
                if "params_ema" in state_dict:
                    state_dict = state_dict["params_ema"]
                elif "params" in state_dict:
                    state_dict = state_dict["params"]

            # Remove module. prefix
            new_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_dict[k[7:]] = v
                else:
                    new_dict[k] = v

            model.load_state_dict(new_dict, strict=True)
            model.to(device).eval()

            # Optimization: channels last
            model = model.to(memory_format=torch.channels_last)

            self.model = model

            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            print(f"      Parameters: {params:,}")

        except Exception as e:
            print(f"[PyTorch] Error loading model: {e}")
            raise

    def infer(self, img_tensor: torch.Tensor, use_amp: bool = False) -> torch.Tensor:
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(img_tensor)
            else:
                return self.model(img_tensor)

    def compile_model(self) -> None:
        """Apply torch.compile to the model."""
        print("      Compiling model (mode='max-autotune', dynamic=True)...")
        # dynamic=True is crucial for benchmarking images of different sizes without recompiling
        self.model = torch.compile(self.model, mode="max-autotune", dynamic=True)



# ---------------------------------------------------------------------
# TENSORRT RUNNER
# ---------------------------------------------------------------------


class TRTRunner:
    def __init__(self, engine_path: str) -> None:
        if not HAS_TRT:
            raise RuntimeError("TensorRT not installed")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.stream = torch.cuda.Stream()

        print(f"[TRT] Loading Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_name = "input"
        self.output_name = "output"

    def infer_with_buffers(
        self, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> torch.Tensor:
        self.context.set_input_shape(self.input_name, input_tensor.shape)
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return output_tensor


# ---------------------------------------------------------------------
# BENCHMARK FUNCTIONS
# ---------------------------------------------------------------------


def benchmark_pytorch(
    runner: PyTorchRunner,
    images: list[Path],
    scale: int,
    use_amp: bool = False,
    device: str = "cuda",
    mode_suffix: str = "",
) -> dict:
    """Benchmark PyTorch model."""
    mode = ("PyTorch FP16" if use_amp else "PyTorch FP32") + mode_suffix
    print(f"\n--- Benchmarking {mode} ---")

    latencies = []
    image_sizes = []

    torch.cuda.reset_peak_memory_stats()

    # Warmup
    dummy = torch.zeros(1, 3, 64, 64, device=device).to(
        memory_format=torch.channels_last
    )
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
        img_t = img_t.div_(255.0).to(memory_format=torch.channels_last)

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
    runner: TRTRunner,
    images: list[Path],
    scale: int,
    device: str = "cuda",
) -> dict:
    """Benchmark TensorRT engine."""
    mode = "TensorRT FP16"
    print(f"\n--- Benchmarking {mode} ---")

    latencies = []
    image_sizes = []

    torch.cuda.reset_peak_memory_stats()

    # Warmup
    dummy = torch.zeros(1, 3, 64, 64, device=device, dtype=torch.float32)
    out_dummy = torch.empty(
        1, 3, 64 * scale, 64 * scale, device=device, dtype=torch.float32
    )
    for _ in range(3):
        runner.infer_with_buffers(dummy, out_dummy)
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

        out_t = torch.empty(
            1, 3, h * scale, w * scale, device=device, dtype=torch.float32
        )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        runner.infer_with_buffers(img_t, out_t)
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


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


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
        pt_runner = PyTorchRunner(
            args.pt_model,
            args.arch,
            args.scale,
            use_content_aware=args.use_content_aware,
            upsampler_alpha=args.upsampler_alpha,
            detail_gain=args.detail_gain,
        )

        # FP32 benchmark
        results.append(benchmark_pytorch(pt_runner, images, args.scale, use_amp=False))
        torch.cuda.empty_cache()

        # FP16 (AMP) benchmark
        results.append(benchmark_pytorch(pt_runner, images, args.scale, use_amp=True))
        torch.cuda.empty_cache()

        # Compiled Benchmark (Future Proofing Speed Check)
        # Note: First run will include compilation time, so we need a warmup
        pt_runner.compile_model()
        results.append(benchmark_pytorch(pt_runner, images, args.scale, use_amp=True, mode_suffix=" (Compiled)"))
        torch.cuda.empty_cache()


        del pt_runner
        torch.cuda.empty_cache()

    # TensorRT Benchmark
    if args.trt_engine:
        if not HAS_TRT:
            print("TensorRT not installed, skipping TRT benchmark.")
        else:
            trt_runner = TRTRunner(args.trt_engine)
            results.append(benchmark_tensorrt(trt_runner, images, args.scale))
            del trt_runner
            torch.cuda.empty_cache()

    # Print markdown table
    if results:
        print_markdown_table(results, args.arch or "unknown", system_info)


if __name__ == "__main__":
    main()
