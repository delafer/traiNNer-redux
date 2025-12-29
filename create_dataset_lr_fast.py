#!/usr/bin/env python3
"""
FAST parallel dataset creation for super-resolution.
Uses multiprocessing to handle large datasets efficiently.

Author: Philip Hofmann
"""

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def process_single_image(args):
    """Process a single image (for multiprocessing)."""
    hr_path, lr_dir, scale, method = args

    try:
        # Load image
        img = Image.open(hr_path).convert("RGB")
        orig_w, orig_h = img.size
        new_w, new_h = orig_w // scale, orig_h // scale

        # Apply downsampling method
        if method == "bicubic_aa":
            lr_img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        elif method == "lanczos":
            lr_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        elif method == "pytorch_bicubic":
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)
            lr_tensor = F.interpolate(
                img_tensor, size=(new_h, new_w), mode="bicubic", align_corners=False
            )
            lr_img = transforms.ToPILImage()(lr_tensor.squeeze(0).clamp(0, 1))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Save
        lr_path = lr_dir / hr_path.name
        lr_img.save(lr_path, quality=95, optimize=True)

        return True, str(hr_path)

    except Exception as e:
        return False, f"{hr_path}: {e!s}"


def get_image_files(hr_dir):
    """Get all image files from directory."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    hr_files = []
    for ext in img_extensions:
        hr_files.extend(hr_dir.glob(f"*{ext}"))
        hr_files.extend(hr_dir.glob(f"*{ext.upper()}"))
    return sorted(hr_files)


def create_lr_dataset_fast(
    hr_dir: Path,
    lr_dir: Path,
    scale: int,
    method: str = "bicubic_aa",
    num_workers: int | None = None,
) -> None:
    """
    Create LR dataset using multiprocessing for speed.

    Args:
        hr_dir: Directory with HR images
        lr_dir: Directory for LR images
        scale: Downsample scale
        method: Degradation method
        num_workers: Number of parallel processes (None = auto-detect)
    """
    # Setup
    lr_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues

    # Get files
    hr_files = get_image_files(hr_dir)
    total_files = len(hr_files)

    if total_files == 0:
        print(f"No image files found in {hr_dir}")
        return

    print("🚀 Fast dataset creation starting...")
    print(f"📁 HR directory: {hr_dir}")
    print(f"📁 LR directory: {lr_dir}")
    print(f"🔢 Total images: {total_files:,}")
    print(f"⚙️  Scale: {scale}x")
    print(f"🎯 Method: {method}")
    print(f"🚀 Workers: {num_workers}")
    print("-" * 60)

    # Prepare arguments for multiprocessing
    args_list = [(hr_path, lr_dir, scale, method) for hr_path in hr_files]

    start_time = time.time()
    processed = 0
    successes = 0
    failures = 0

    # Process with progress tracking
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_image, args): args[0] for args in args_list
        }

        for future in as_completed(futures):
            success, result = future.result()

            if success:
                successes += 1
            else:
                failures += 1
                print(f"❌ Error: {result}")

            processed += 1

            # Progress update
            if processed % 100 == 0 or processed == total_files:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (total_files - processed) / rate if rate > 0 else 0

                print(
                    f"📊 Progress: {processed:,}/{total_files:,} "
                    f"({100 * processed / total_files:.1f}%) | "
                    f"✅ {successes} ❌ {failures} | "
                    f"🚀 {rate:.1f} img/s | "
                    f"⏱️  ETA: {eta / 60:.1f} min"
                )

    # Final summary
    total_time = time.time() - start_time
    final_rate = successes / total_time

    print("-" * 60)
    print("🎉 Dataset creation COMPLETE!")
    print(f"✅ Successfully processed: {successes:,} images")
    print(f"❌ Failed: {failures:,} images")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print(f"🚀 Average speed: {final_rate:.1f} images/second")
    print(f"📁 LR dataset saved to: {lr_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", required=True, help="Directory with HR images")
    parser.add_argument("--lr_dir", required=True, help="Directory for LR images")
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4], help="Downsample scale"
    )
    parser.add_argument(
        "--method",
        default="bicubic_aa",
        choices=["bicubic_aa", "lanczos", "pytorch_bicubic"],
        help="Degradation method",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers"
    )

    args = parser.parse_args()

    hr_dir = Path(args.hr_dir)
    lr_dir = Path(args.lr_dir)

    if not hr_dir.exists():
        raise FileNotFoundError(f"HR directory not found: {hr_dir}")

    create_lr_dataset_fast(hr_dir, lr_dir, args.scale, args.method, args.workers)
