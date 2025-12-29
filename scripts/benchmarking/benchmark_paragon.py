#!/usr/bin/env python3
"""
ParagonSR Comprehensive Benchmarking Tool
Author: Philip Hofmann

Description:
Benchmarks all ParagonSR variants (tiny, xs, s, m, l, xl) across different scales
and model formats to measure inference speed and VRAM usage.

Usage:
python3 scripts/benchmarking/benchmark_paragon.py --models_dir /path/to/models --images_dir /path/to/test_images --output results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import GPUtil
import numpy as np
import onnxruntime as ort
import psutil
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from safetensors.torch import load_file
from traiNNer.archs.paragonsr_arch import (
    paragonsr_l,
    paragonsr_m,
    paragonsr_s,
    paragonsr_tiny,
    paragonsr_xl,
    paragonsr_xs,
)


class PerformanceMonitor:
    """Monitor system performance during inference."""

    def __init__(self) -> None:
        self.gpu_initialized = False
        try:
            self.gpus = GPUtil.getGPUs()
            self.gpu_initialized = len(self.gpus) > 0
        except:
            self.gpu_initialized = False

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage."""
        memory = {}

        # CPU Memory
        memory["cpu_percent"] = psutil.virtual_memory().percent
        memory["cpu_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # GPU Memory
        if self.gpu_initialized:
            gpu = self.gpus[0]  # Assume single GPU for now
            memory["gpu_percent"] = (gpu.memoryUsed / gpu.memoryTotal) * 100
            memory["gpu_used_mb"] = gpu.memoryUsed
            memory["gpu_total_mb"] = gpu.memoryTotal
        else:
            memory["gpu_percent"] = 0.0
            memory["gpu_used_mb"] = 0.0
            memory["gpu_total_mb"] = 0.0

        return memory


class ParagonBenchmark:
    """Main benchmarking class for ParagonSR models."""

    def __init__(self, models_dir: str, images_dir: str, output_file: str) -> None:
        self.models_dir = Path(models_dir)
        self.images_dir = Path(images_dir)
        self.output_file = output_file
        self.monitor = PerformanceMonitor()

        # Model variants
        self.variants = {
            "tiny": paragonsr_tiny,
            "xs": paragonsr_xs,
            "s": paragonsr_s,
            "m": paragonsr_m,
            "l": paragonsr_l,
            "xl": paragonsr_xl,
        }

        # Benchmark results
        self.results = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda
                if torch.cuda.is_available()
                else None,
            },
            "models": {},
        }

    def load_test_images(
        self, num_images: int = 50, target_size: int = 512
    ) -> list[torch.Tensor]:
        """Load and preprocess test images."""
        images = []
        image_files = list(self.images_dir.glob("*.png")) + list(
            self.images_dir.glob("*.jpg")
        )

        if len(image_files) < num_images:
            print(
                f"Warning: Only found {len(image_files)} images, expected {num_images}"
            )
            num_images = len(image_files)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        for _i, img_path in enumerate(image_files[:num_images]):
            try:
                img = Image.open(img_path).convert("RGB")

                # Resize to target size
                img = img.resize((target_size, target_size), Image.BICUBIC)

                # Convert to tensor and normalize
                tensor = transform(img)
                images.append(tensor)

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        return images

    def load_pytorch_model(
        self, variant: str, scale: int, model_path: str
    ) -> torch.nn.Module:
        """Load PyTorch model (fused)."""
        model_func = self.variants[variant]
        model = model_func(scale=scale)
        model.eval()

        # Load fused weights
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)

        return model

    def load_onnx_model(self, model_path: str) -> ort.InferenceSession:
        """Load ONNX model."""
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(model_path, providers=providers)
        return session

    def benchmark_pytorch_model(
        self,
        model: torch.nn.Module,
        test_images: list[torch.Tensor],
        model_name: str,
        num_warmup: int = 5,
    ) -> dict:
        """Benchmark PyTorch model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Warm-up
        for _i in range(num_warmup):
            with torch.no_grad():
                _ = model(test_images[0].unsqueeze(0).to(device))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times = []
        memory_peak = {"cpu_percent": 0, "gpu_percent": 0, "gpu_used_mb": 0}

        for img in test_images:
            # Clear GPU cache for accurate measurement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            start_memory = self.monitor.get_memory_usage()

            start_time = time.time()
            with torch.no_grad():
                _ = model(img.unsqueeze(0).to(device))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            end_memory = self.monitor.get_memory_usage()

            # Track peak memory
            memory_peak["cpu_percent"] = max(
                memory_peak["cpu_percent"], end_memory["cpu_percent"]
            )
            memory_peak["gpu_percent"] = max(
                memory_peak["gpu_percent"], end_memory["gpu_percent"]
            )
            memory_peak["gpu_used_mb"] = max(
                memory_peak["gpu_used_mb"], end_memory["gpu_used_mb"]
            )

            inference_time = end_time - start_time
            times.append(inference_time)

        return {
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "memory_peak": memory_peak,
            "device": str(device),
            "warmup_runs": num_warmup,
            "num_inferences": len(times),
        }

    def benchmark_onnx_model(
        self,
        session: ort.InferenceSession,
        test_images: list[torch.Tensor],
        model_name: str,
        num_warmup: int = 5,
    ) -> dict:
        """Benchmark ONNX model."""

        # Preprocess images for ONNX (denormalize)
        def preprocess_for_onnx(img_tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            # Convert to numpy and ensure correct shape
            return img_denorm.numpy()

        # Warm-up
        for _i in range(num_warmup):
            input_data = preprocess_for_onnx(test_images[0])
            _ = session.run(None, {"input": input_data})

        # Benchmark
        times = []
        memory_peak = {"cpu_percent": 0, "gpu_percent": 0, "gpu_used_mb": 0}

        for img in test_images:
            start_memory = self.monitor.get_memory_usage()

            start_time = time.time()
            input_data = preprocess_for_onnx(img)
            _ = session.run(None, {"input": input_data})
            end_time = time.time()

            end_memory = self.monitor.get_memory_usage()

            # Track peak memory
            memory_peak["cpu_percent"] = max(
                memory_peak["cpu_percent"], end_memory["cpu_percent"]
            )
            memory_peak["gpu_percent"] = max(
                memory_peak["gpu_percent"], end_memory["gpu_percent"]
            )
            memory_peak["gpu_used_mb"] = max(
                memory_peak["gpu_used_mb"], end_memory["gpu_used_mb"]
            )

            inference_time = end_time - start_time
            times.append(inference_time)

        # Get model info
        input_info = session.get_inputs()[0]

        return {
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "memory_peak": memory_peak,
            "input_shape": input_info.shape,
            "input_type": input_info.type,
            "providers": session.get_providers(),
            "warmup_runs": num_warmup,
            "num_inferences": len(times),
        }

    def find_model_files(self, variant: str, scale: int) -> dict[str, str | None]:
        """Find model files for a given variant and scale."""
        files = {
            "fused_safetensors": None,
            "fp32_onnx": None,
            "fp16_onnx": None,
        }

        # Look for fused safetensors
        fused_patterns = [
            f"*paragon*{variant}*{scale}x*_fused.safetensors",
            f"*Paragon*{variant.upper()}*{scale}x*_fused.safetensors",
            f"{variant}_{scale}x_fused.safetensors",
        ]

        for pattern in fused_patterns:
            found = list(self.models_dir.glob(pattern))
            if found:
                files["fused_safetensors"] = str(found[0])
                break

        # Look for ONNX files
        onnx_patterns = {
            "fp32_onnx": [
                f"*paragon*{variant}*{scale}x*fp32.onnx",
                f"*Paragon*{variant.upper()}*{scale}x*fp32.onnx",
            ],
            "fp16_onnx": [
                f"*paragon*{variant}*{scale}x*fp16.onnx",
                f"*Paragon*{variant.upper()}*{scale}x*fp16.onnx",
            ],
        }

        for file_type, patterns in onnx_patterns.items():
            for pattern in patterns:
                found = list(self.models_dir.glob(pattern))
                if found:
                    files[file_type] = str(found[0])
                    break

        return files

    def run_benchmark(
        self,
        variants: list[str] | None = None,
        scales: list[int] | None = None,
        num_images: int = 50,
    ) -> None:
        """Run comprehensive benchmark."""
        if variants is None:
            variants = list(self.variants.keys())
        if scales is None:
            scales = [2, 4]

        print("🚀 Starting ParagonSR Benchmark")
        print(f"   Variants: {variants}")
        print(f"   Scales: {scales}x")
        print(f"   Images: {num_images}")
        print(f"   Models dir: {self.models_dir}")
        print(f"   Images dir: {self.images_dir}")
        print("=" * 60)

        # Load test images
        print("📷 Loading test images...")
        test_images = self.load_test_images(num_images)
        print(f"✅ Loaded {len(test_images)} test images")

        # Benchmark each variant and scale
        for variant in variants:
            if variant not in self.variants:
                print(f"⚠️  Unknown variant: {variant}")
                continue

            for scale in scales:
                model_key = f"{variant}_{scale}x"
                print(f"\n🏗️  Benchmarking ParagonSR-{variant.upper()} {scale}x")

                # Find model files
                model_files = self.find_model_files(variant, scale)
                variant_results = {
                    "variant": variant,
                    "scale": scale,
                    "files_found": {k: v is not None for k, v in model_files.items()},
                    "formats": {},
                }

                print(
                    f"   Found files: {[(k, v is not None) for k, v in model_files.items()]}"
                )

                # Benchmark each available format
                for format_name, file_path in model_files.items():
                    if file_path is None or not os.path.exists(file_path):
                        print(f"   ⚠️  {format_name}: File not found")
                        continue

                    print(f"   🔄 Benchmarking {format_name}...")

                    try:
                        if format_name == "fused_safetensors":
                            # PyTorch benchmark
                            model = self.load_pytorch_model(variant, scale, file_path)
                            result = self.benchmark_pytorch_model(
                                model, test_images, model_key
                            )

                        elif format_name.endswith("_onnx"):
                            # ONNX benchmark
                            session = self.load_onnx_model(file_path)
                            result = self.benchmark_onnx_model(
                                session, test_images, model_key
                            )

                        variant_results["formats"][format_name] = result
                        print(f"   ✅ {format_name}: {result['avg_time_ms']:.2f}ms avg")

                    except Exception as e:
                        print(f"   ❌ {format_name}: Benchmark failed - {e}")
                        variant_results["formats"][format_name] = {"error": str(e)}

                # Store results
                self.results["models"][model_key] = variant_results

        # Save results
        print(f"\n💾 Saving results to {self.output_file}")
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print("\n🎉 Benchmark completed!")
        self.print_summary()

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 60)

        for model_key, model_data in self.results["models"].items():
            print(f"\n🏗️  {model_key.upper()}")
            for format_name, result in model_data["formats"].items():
                if "error" in result:
                    print(f"   {format_name:15}: ERROR - {result['error']}")
                else:
                    avg_time = result["avg_time_ms"]
                    gpu_mem = result["memory_peak"]["gpu_used_mb"]
                    print(
                        f"   {format_name:15}: {avg_time:6.2f}ms avg, {gpu_mem:6.0f}MB VRAM"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ParagonSR Comprehensive Benchmarking")
    parser.add_argument(
        "--models_dir", required=True, help="Directory containing model files"
    )
    parser.add_argument(
        "--images_dir", required=True, help="Directory containing test images"
    )
    parser.add_argument(
        "--output", default="benchmark_results.json", help="Output results file"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["tiny", "xs", "s", "m", "l", "xl"],
        help="Variants to benchmark",
    )
    parser.add_argument(
        "--scales", nargs="+", type=int, default=[2, 4], help="Scales to benchmark"
    )
    parser.add_argument(
        "--num_images", type=int, default=50, help="Number of test images to use"
    )

    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.models_dir):
        print(f"❌ Models directory not found: {args.models_dir}")
        sys.exit(1)

    if not os.path.exists(args.images_dir):
        print(f"❌ Images directory not found: {args.images_dir}")
        sys.exit(1)

    # Run benchmark
    benchmark = ParagonBenchmark(args.models_dir, args.images_dir, args.output)
    benchmark.run_benchmark(args.variants, args.scales, args.num_images)


if __name__ == "__main__":
    main()
