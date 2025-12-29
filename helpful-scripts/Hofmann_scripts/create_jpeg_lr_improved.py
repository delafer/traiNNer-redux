#!/usr/bin/env python3
"""
Improved JPEG Artifact Dataset Generator for De-JPEG Super-Resolution Training

Description:
Creates JPEG-compressed LR images from clean LR images for training de-JPEG SR models.
Supports both standard (downsample→JPEG) and reverse (JPEG→downsample) degradation
paths to better match real-world scenarios.

Features:
-   **Realistic Quality Ranges**: Default Q60-95 matches real-world web photos
-   **Dual Degradation Paths**:
    - Standard (70%): LR → JPEG compress → LR_JPG
    - Reverse (30%): LR → JPEG compress → upsample → downsample → LR_JPG
      (simulates JPEG from camera, then browser/app downsampling)
-   **Configurable Quality**: Preset modes (realistic, severe, extreme) or custom range
-   **High Performance**: Multi-threaded processing with progress bar
-   **PNG Output**: Losslessly preserves artifacts for training

Usage:
    # Realistic web photos (recommended)
    python create_jpeg_lr_improved.py --input lr_x2 --output lr_x2_jpg --preset realistic

    # Severe artifacts for stress-testing
    python create_jpeg_lr_improved.py --input lr_x2 --output lr_x2_jpg --preset severe

    # Custom quality range
    python create_jpeg_lr_improved.py --input lr_x2 --output lr_x2_jpg --q_min 50 --q_max 95

    # Disable reverse degradation path (faster, less diverse)
    python create_jpeg_lr_improved.py --input lr_x2 --output lr_x2_jpg --no_reverse

Dependencies:
    pip install opencv-python numpy tqdm
"""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Quality presets
PRESETS = {
    "realistic": (70, 95),  # Real-world web photos, social media
    "moderate": (60, 95),  # Broader range, still realistic
    "severe": (40, 95),  # Include heavily compressed images
    "extreme": (30, 95),  # Stress-test (rare in real world)
}


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """
    Applies JPEG compression to an image in memory.

    Args:
        img: Input image as NumPy array (BGR format)
        quality: JPEG quality (0-100, higher = better quality)

    Returns:
        JPEG-compressed image as NumPy array
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_buffer = cv2.imencode(".jpg", img, encode_param)

    if not success:
        raise RuntimeError("JPEG encoding failed")

    return cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)


def degrade_image_standard(img: np.ndarray, quality: int) -> np.ndarray:
    """
    Standard degradation: Direct JPEG compression.

    Simulates: photo downsampled for web → JPEG compressed
    """
    return jpeg_compress(img, quality)


def degrade_image_reverse(img: np.ndarray, quality: int, scale: int = 2) -> np.ndarray:
    """
    Reverse degradation: JPEG compress → upsample → downsample.

    Simulates: JPEG photo from camera → browser/app downsampling
    This creates different artifact patterns where JPEG blocks interact with
    downsampling, producing more complex degradations.

    Args:
        img: Input LR image
        quality: JPEG quality for compression
        scale: Upsampling factor (typically 2 for 2x SR)
    """
    h, w = img.shape[:2]

    # Apply lighter JPEG compression first
    # Use slightly higher quality since this represents camera JPEG at higher res
    camera_quality = min(quality + 10, 98)  # Camera typically uses higher Q
    img_jpeg = jpeg_compress(img, camera_quality)

    # Upsample to simulate viewing at higher resolution
    img_up = cv2.resize(img_jpeg, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Downsample back to original LR size (simulates browser/app downsampling)
    # Use different interpolation to create variety
    downsample_method = random.choice(
        [
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
        ]
    )
    img_down = cv2.resize(img_up, (w, h), interpolation=downsample_method)

    # Apply final JPEG compression at target quality
    return jpeg_compress(img_down, quality)


def process_image(file_info: tuple[Path, Path, int, int, bool, int]) -> None:
    """
    Reads an image, applies JPEG degradation, and saves as PNG.

    Args:
        file_info: Tuple of (input_path, output_path, min_q, max_q, enable_reverse, scale)
    """
    input_path, output_path, min_q, max_q, enable_reverse, scale = file_info

    try:
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Warning: Could not read {input_path.name}")
            return

        # Determine random quality
        quality = random.randint(min_q, max_q)

        # Choose degradation path
        if enable_reverse:
            # 70% standard, 30% reverse degradation
            use_reverse = random.random() < 0.3
        else:
            use_reverse = False

        # Apply degradation
        if use_reverse:
            degraded_img = degrade_image_reverse(img, quality, scale)
        else:
            degraded_img = degrade_image_standard(img, quality)

        # Save as PNG (lossless storage of artifacts)
        save_path = output_path.with_suffix(".png")
        cv2.imwrite(str(save_path), degraded_img)

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create JPEG-Artifact Dataset for De-JPEG Super-Resolution Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Realistic web photos (recommended for most cases)
  python %(prog)s --input lr_x2 --output lr_x2_jpg --preset realistic

  # Severe artifacts for robust training
  python %(prog)s --input lr_x2 --output lr_x2_jpg --preset severe

  # Custom quality range with 4x scale
  python %(prog)s --input lr_x4 --output lr_x4_jpg --q_min 50 --q_max 95 --scale 4

Quality Presets:
  realistic: Q70-95 (web photos, social media - recommended)
  moderate:  Q60-95 (broader range, still realistic)
  severe:    Q40-95 (heavily compressed images)
  extreme:   Q30-95 (stress-test, rare in real world)
        """,
    )

    parser.add_argument("--input", required=True, help="Path to clean LR folder")
    parser.add_argument(
        "--output", required=True, help="Path to save JPEG-degraded LR images"
    )

    # Quality settings
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="moderate",
        help="Quality preset (default: moderate)",
    )
    quality_group.add_argument(
        "--q_min",
        type=int,
        metavar="Q",
        help="Custom minimum JPEG quality (0-100, overrides preset)",
    )

    parser.add_argument(
        "--q_max",
        type=int,
        default=95,
        metavar="Q",
        help="Maximum JPEG quality (0-100, default: 95)",
    )

    # Degradation options
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="SR scale factor for reverse degradation (default: 2)",
    )
    parser.add_argument(
        "--no_reverse",
        action="store_true",
        help="Disable reverse degradation path (faster but less diverse)",
    )

    # Performance
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of CPU threads (default: 8)"
    )

    args = parser.parse_args()

    # Determine quality range
    if args.q_min is not None:
        q_min = args.q_min
        q_max = args.q_max
        quality_desc = f"Custom (Q{q_min}-{q_max})"
    else:
        q_min, q_max = PRESETS[args.preset]
        quality_desc = f"{args.preset.capitalize()} preset (Q{q_min}-{q_max})"

    # Validate quality range
    if not (0 <= q_min <= 100 and 0 <= q_max <= 100):
        print("Error: Quality values must be in range 0-100")
        return
    if q_min > q_max:
        print("Error: q_min must be <= q_max")
        return

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: Input folder not found at '{input_dir}'")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    files = sorted([f for f in input_dir.glob("*") if f.suffix.lower() in valid_exts])

    if not files:
        print(f"No images found in '{input_dir}'")
        return

    # Display settings
    print("=" * 60)
    print("JPEG Artifact Dataset Generator")
    print("=" * 60)
    print(f"Input folder:  {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Images found:  {len(files)}")
    print(f"Quality range: {quality_desc}")
    print(f"SR scale:      {args.scale}x")
    print(
        f"Reverse path:  {'Enabled (30% of images)' if not args.no_reverse else 'Disabled'}"
    )
    print(f"Workers:       {args.workers}")
    print("=" * 60)

    # Prepare work items
    tasks = []
    for f in files:
        out_file = output_dir / f.name
        tasks.append((f, out_file, q_min, q_max, not args.no_reverse, args.scale))

    # Process in parallel with progress bar
    print("\nApplying JPEG compression...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(
            tqdm(
                executor.map(process_image, tasks),
                total=len(tasks),
                unit="img",
                desc="Processing",
            )
        )

    print("\n" + "=" * 60)
    print("✓ Dataset creation complete!")
    print(f"✓ JPEG-degraded images saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
