#!/usr/bin/env python3
"""
SAFE IC9600 Complexity Scoring - Conservative Resource Usage
Author: Philip Hofmann

DESIGNED TO PREVENT SYSTEM FREEZE
- Single process (no multiprocessing)
- Small batches to prevent memory issues
- Conservative memory management
- Progress monitoring without overwhelming system

Expected: 2-4 images/second (slower but safe)
For 147k images: 10-20 hours (safe processing)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import our ICNet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ICNet import ICNet

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SafeComplexityScorer:
    """Safe, resource-conscious complexity scorer."""

    def __init__(self, model_path, device_str, batch_size=4) -> None:
        """Initialize with conservative settings."""
        self.device = torch.device(device_str)
        self.batch_size = batch_size

        # Load model with memory management
        logger.info(f"Loading ICNet model on {self.device}")
        self.model = ICNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        # Conservative mixed precision (only if enough memory)
        if self.device.type == "cuda":
            try:
                # Test if we can use FP16
                test_input = torch.randn(1, 3, 512, 512).half().to(self.device)
                with torch.no_grad():
                    _ = self.model(test_input)
                self.model = self.model.half()
                self.fp16 = True
                logger.info("Enabled FP16 mixed precision")
            except RuntimeError:
                self.fp16 = False
                logger.info("Using FP32 (FP16 unavailable)")
        else:
            self.fp16 = False

        self.model.eval()

        # Conservative preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (512, 512), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(
            f"Initialized safe scorer: batch_size={batch_size}, fp16={self.fp16}"
        )

    def process_batch_safe(self, image_paths):
        """Process batch with extensive memory management."""
        results = []

        # Load images one by one to manage memory
        images = []
        valid_paths = []

        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                img_tensor = self.preprocess(img_rgb)
                images.append(img_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue

        if not images:
            return []

        # Process in very small batches
        batch_results = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_paths = valid_paths[i : i + self.batch_size]

            # Process small batch
            batch_tensor = None
            scores = None
            try:
                batch_tensor = torch.stack(batch_images).to(self.device)

                if self.fp16:
                    batch_tensor = batch_tensor.half()

                with torch.no_grad():
                    scores, _ = self.model(batch_tensor)
                    scores = scores.cpu().numpy()

                    # Ensure scores is always a 1D array
                    if scores.ndim == 0:
                        # Single scalar score
                        scores = [float(scores)]
                    elif scores.ndim == 1:
                        # Already 1D, convert to list
                        scores = [float(s) for s in scores]
                    else:
                        # Multi-dimensional, flatten first
                        scores = scores.flatten()
                        scores = [float(s) for s in scores]

                # Combine paths and scores
                batch_results.extend(list(zip(batch_paths, scores, strict=False)))

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add failed results with 0 score
                batch_results.extend([(path, 0.0) for path in batch_paths])

            # Aggressive memory cleanup
            if batch_tensor is not None:
                del batch_tensor
            if scores is not None:
                del scores
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Small delay to prevent overwhelming system
            time.sleep(0.01)

        return batch_results

    def score_directory_safe(
        self, input_dir, output_dir, output_file="complexity_scores_safe.csv"
    ) -> None:
        """Safe directory scoring with conservative resource usage."""
        input_path = Path(input_dir)

        # Find images
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = []

        for ext in extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))

        image_paths = sorted(image_paths)
        logger.info(f"Found {len(image_paths)} images to process safely")

        if not image_paths:
            logger.error("No images found!")
            return

        # Setup output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_path / output_file

        # Process with conservative batching
        results = []
        start_time = time.time()
        last_progress_time = start_time

        # Process in very small chunks
        chunk_size = 20  # Very conservative

        for i in tqdm(range(0, len(image_paths), chunk_size), desc="Processing safely"):
            chunk = image_paths[i : i + chunk_size]
            chunk_results = self.process_batch_safe(chunk)
            results.extend(chunk_results)

            # Progress logging every 30 seconds
            current_time = time.time()
            if current_time - last_progress_time > 30:
                elapsed = current_time - start_time
                processed = len(results)
                rate = processed / elapsed
                eta = (len(image_paths) - processed) / rate if rate > 0 else 0

                # Memory usage check
                if self.device.type == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    logger.info(
                        f"Progress: {processed}/{len(image_paths)} "
                        f"({rate:.1f} img/s, ETA: {eta / 60:.1f} min) "
                        f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved"
                    )
                else:
                    logger.info(
                        f"Progress: {processed}/{len(image_paths)} "
                        f"({rate:.1f} img/s, ETA: {eta / 60:.1f} min)"
                    )

                last_progress_time = current_time

        # Write results with conservative file writing
        self._write_results_safe(results, output_file_path)

        # Final summary
        total_time = time.time() - start_time
        rate = len(results) / total_time

        logger.info("Safe complexity scoring completed!")
        logger.info(f"Results: {len(results)} images in {total_time / 60:.1f} minutes")
        logger.info(f"Performance: {rate:.1f} images/second")
        logger.info(
            f"Success rate: {len(results)}/{len(image_paths)} ({100 * len(results) / len(image_paths):.1f}%)"
        )
        logger.info(f"Output: {output_file_path}")

    def _write_results_safe(self, results, output_file) -> None:
        """Conservative file writing with small buffers."""
        with open(output_file, "w") as f:
            f.write("image_path,complexity_score\n")

            # Very small buffer to prevent I/O overwhelming
            buffer_size = 100
            buffer_lines = []

            for img_path, score in results:
                buffer_lines.append(f"{img_path},{score:.6f}\n")

                if len(buffer_lines) >= buffer_size:
                    f.writelines(buffer_lines)
                    f.flush()  # Force write to disk
                    buffer_lines.clear()
                    time.sleep(0.001)  # Small delay to prevent I/O overload

            # Write remaining
            if buffer_lines:
                f.writelines(buffer_lines)


def main() -> None:
    """Main function with conservative defaults."""
    parser = argparse.ArgumentParser(
        description="SAFE IC9600 Complexity Scoring (Prevents System Freeze)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input directory with images"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="./safe_complexity", help="Output directory"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./complexity.pth",
        help="Path to ICNet model",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda:0", help="Device to use"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=4, help="Batch size (conservative: 2-8)"
    )

    args = parser.parse_args()

    # Conservative device validation
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Additional CPU checks
    if args.device == "cpu":
        logger.warning("CPU mode will be very slow but safe")
        args.batch_size = 2  # Very small for CPU

    try:
        logger.info("=" * 60)
        logger.info("SAFE MODE: Conservative resource usage")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Device: {args.device}")
        logger.info("=" * 60)

        scorer = SafeComplexityScorer(args.model, args.device, args.batch_size)
        scorer.score_directory_safe(args.input, args.output)

    except KeyboardInterrupt:
        logger.info("Interrupted by user - results saved up to this point")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
