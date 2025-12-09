import argparse
import sys
from pathlib import Path

# Remove unused imports
# import time
# from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=""):
        return iterable  # type: ignore


import cv2
import numpy as np  # type: ignore
import torch
import torch.nn.functional as F  # type: ignore

# Try importing TensorRT
try:
    import tensorrt as trt  # type: ignore

    HAS_TRT = True
except ImportError:
    HAS_TRT = False

# ---------------------------------------------------------------------
# PYTORCH RUNNER
# ---------------------------------------------------------------------


class PyTorchRunner:
    def __init__(
        self, model_path: str, arch: str, scale: int, device: str = "cuda"
    ) -> None:
        self.device = device
        self.scale = scale
        print(f"[PyTorch] Loading {arch} from {model_path}...")

        # Helper to load arch
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import paragonsr2_arch  # type: ignore

            # Map string to class
            arch_map = {
                "paragonsr2_realtime": paragonsr2_arch.paragonsr2_realtime,
                "paragonsr2_stream": paragonsr2_arch.paragonsr2_stream,
                "paragonsr2_photo": paragonsr2_arch.paragonsr2_photo,
                "paragonsr2_pro": paragonsr2_arch.paragonsr2_pro,
            }
            if arch not in arch_map:
                raise ValueError(f"Unknown arch: {arch}")

            model = arch_map[arch](scale=scale)

            if str(model_path).endswith(".safetensors"):
                from safetensors.torch import load_file  # type: ignore

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

            # Optimization: fuse if available
            if hasattr(model, "fuse_for_release"):
                model.fuse_for_release()

            # Optimization: channels last
            model = model.to(memory_format=torch.channels_last)

            self.model = model

        except Exception as e:
            print(f"[PyTorch] Error loading model: {e}")
            sys.exit(1)

    def infer(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(img_tensor)


# ---------------------------------------------------------------------
# TENSORRT RUNNER
# ---------------------------------------------------------------------


class TRTRunner:
    def __init__(self, engine_path: str) -> None:
        if not HAS_TRT:
            print("[TRT] TensorRT not installed, skipping.")
            sys.exit(1)

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.stream = torch.cuda.Stream()

        print(f"[TRT] Loading Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_name = "input"
        self.output_name = "output"

    def infer(self, img_tensor: torch.Tensor) -> None:
        # Not utilized
        pass

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
# BENCHMARK SUITE
# ---------------------------------------------------------------------


def benchmark(
    runner: PyTorchRunner | TRTRunner,
    mode: str,
    images: list[Path],
    scale: int,
    device: str = "cuda",
) -> tuple[float, float, float]:
    print(f"\n--- Benchmarking {mode} ---")

    # Metrics
    latencies = []

    torch.cuda.reset_peak_memory_stats()
    # start_vram = torch.cuda.memory_allocated() / (1024**3)

    # Warmup
    dummy = torch.zeros(1, 3, 64, 64, device=device).to(
        memory_format=torch.channels_last
    )
    if mode == "PyTorch":
        runner.infer(dummy)
    else:
        out_dummy = torch.empty(
            1, 3, 64 * scale, 64 * scale, device=device, dtype=torch.float32
        )
        runner.infer_with_buffers(dummy, out_dummy)  # type: ignore

    torch.cuda.synchronize()

    for img_path in tqdm(images, desc=mode):
        # Load & Preprocess
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _c = img.shape

        img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.div_(255.0).to(memory_format=torch.channels_last)

        # Create Output Buffer (for TRT)
        out_t_trt = None
        if mode == "TensorRT":
            out_t_trt = torch.empty(
                1, 3, int(h * scale), int(w * scale), device=device, dtype=torch.float32
            )

        # Reset stats for THIS run
        # Note: VRAM is cumulative usually, but we check peak since start

        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        if mode == "PyTorch":
            _ = runner.infer(img_t)
        else:
            _ = runner.infer_with_buffers(img_t, out_t_trt)  # type: ignore

        end_event.record()
        torch.cuda.synchronize()

        latencies.append(start_event.elapsed_time(end_event))  # ms

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency

    print(f"Results for {mode}:")
    print(f"  Avg Latency:  {avg_latency:.2f} ms")
    print(f"  Throughput:   {fps:.2f} FPS")
    print(f"  Peak VRAM:    {peak_vram:.3f} GB")

    return avg_latency, fps, peak_vram


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

    args = parser.parse_args()

    images = sorted(Path(args.input).glob("*"))
    images = [p for p in images if p.suffix.lower() in [".jpg", ".png", ".webp"]]

    if not images:
        print("No images found.")
        return

    print(f"Benchmarking on {len(images)} images (Scale: {args.scale}x)")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. PyTorch Benchmark
    if args.pt_model and args.arch:
        pt_runner = PyTorchRunner(args.pt_model, args.arch, args.scale)
        benchmark(pt_runner, "PyTorch", images, args.scale)
        del pt_runner
        torch.cuda.empty_cache()

    # 2. TensorRT Benchmark
    if args.trt_engine:
        if not HAS_TRT:
            print("TensorRT not installed, skipping TRT benchmark.")
        else:
            trt_runner = TRTRunner(args.trt_engine)
            benchmark(trt_runner, "TensorRT", images, args.scale)


if __name__ == "__main__":
    main()
