#!/usr/bin/env python3
"""
Video Compression Degradation Script for Super-Resolution Training

Creates video-compressed LR images from clean LR images. Uses FFmpeg to encode
to H.264/H.265 and extract frames, simulating real-world video compression.

Video compression differs from JPEG:
- Uses temporal prediction (P/B-frames) creating unique artifacts
- 8x8 DCT blocks (H.264) or larger CTUs (H.265)
- In-loop deblocking filter
- Motion compensation artifacts
- Banding in smooth gradients

Usage:
    python create_video_lr.py --input lr_x2 --output lr_x2_video --preset medium

Dependencies:
    pip install opencv-python numpy tqdm
    Requires: ffmpeg (system package)
"""

import argparse
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def check_ffmpeg() -> bool | None:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def apply_video_compression(
    img: np.ndarray,
    codec: str = "libx264",
    crf: int = 28,
    preset: str = "medium",
) -> np.ndarray:
    """
    Apply video compression to a single image using FFmpeg.

    Args:
        img: Input image as numpy array (BGR)
        codec: Video codec (libx264, libx265)
        crf: Constant Rate Factor (0-51, lower=better quality)
               Typical: 18=high, 23=default, 28=medium, 35=low
        preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)

    Returns:
        Compressed image as numpy array
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.png"
        output_path = tmpdir / "output.mp4"
        decoded_path = tmpdir / "decoded.png"

        # Save input
        cv2.imwrite(str(input_path), img)

        # Encode to video (1-frame video)
        encode_cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i",
            str(input_path),
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",  # Standard video format
            "-frames:v",
            "1",  # Single frame
            "-loglevel",
            "error",
            str(output_path),
        ]

        subprocess.run(encode_cmd, check=True, capture_output=True)

        # Decode back to image
        decode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(output_path),
            "-frames:v",
            "1",
            "-loglevel",
            "error",
            str(decoded_path),
        ]

        subprocess.run(decode_cmd, check=True, capture_output=True)

        # Read compressed result
        return cv2.imread(str(decoded_path))


def process_image(file_info) -> None:
    """Process a single image with video compression."""
    input_path, output_path, codec, crf_min, crf_max, preset = file_info

    try:
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Warning: Could not read {input_path.name}")
            return

        # Random CRF within range
        crf = random.randint(crf_min, crf_max)

        # Apply video compression
        compressed_img = apply_video_compression(img, codec, crf, preset)

        # Save as PNG (lossless storage of compression artifacts)
        save_path = output_path.with_suffix(".png")
        cv2.imwrite(str(save_path), compressed_img)

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create video-compressed dataset for SR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a SUBSET (recommended for large datasets!)
  python %(prog)s --input lr_x2 --output lr_x2_video --max_images 5000 --preset medium

  # Medium quality (YouTube-like)
  python %(prog)s --input lr_x2 --output lr_x2_video --preset medium

  # Low quality (streaming, low bitrate)
  python %(prog)s --input lr_x2 --output lr_x2_video --preset fast --crf_min 30 --crf_max 40

  # High quality (Blu-ray-like)
  python %(prog)s --input lr_x2 --output lr_x2_video --preset slow --crf_min 18 --crf_max 25

CRF Values:
  18-22: High quality (Blu-ray, high bitrate streaming)
  23-28: Medium quality (YouTube HD, Netflix)
  29-35: Low quality (streaming, mobile)
  36-51: Very low quality (stress test)

Presets (speed/quality tradeoff):
  ultrafast/fast: Faster encoding, lower quality
  medium: Balanced (default)
  slow/veryslow: Better quality, slower encoding

IMPORTANT for large datasets (100k+ images):
  Use --max_images to limit processing (5k-10k is usually enough for training!)
  Full dataset processing can take DAYS. A subset is faster and often sufficient.
        """,
    )

    parser.add_argument("--input", required=True, help="Path to clean LR folder")
    parser.add_argument(
        "--output", required=True, help="Path to save video-compressed LR images"
    )

    # Codec settings
    parser.add_argument(
        "--codec",
        choices=["libx264", "libx265"],
        default="libx264",
        help="Video codec (default: libx264/H.264)",
    )
    parser.add_argument(
        "--crf_min",
        type=int,
        default=23,
        help="Minimum CRF (lower=better quality, default: 23)",
    )
    parser.add_argument(
        "--crf_max",
        type=int,
        default=35,
        help="Maximum CRF (default: 35)",
    )
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        default="medium",
        help="Encoding preset (default: medium)",
    )

    # Performance
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads (default: 4)"
    )

    # NEW: Subset processing
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all). RECOMMENDED for large datasets!",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle images before processing (useful with --max_images)",
    )

    args = parser.parse_args()

    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/")
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

    # Shuffle if requested
    if args.shuffle:
        import random

        random.shuffle(files)

    # Limit to max_images
    total_files = len(files)
    if args.max_images and args.max_images < total_files:
        files = files[: args.max_images]
        print(
            f"⚠️  Processing only {args.max_images} of {total_files} images (subset mode)"
        )

    # Display settings
    print("=" * 60)
    print("Video Compression Degradation")
    print("=" * 60)
    print(f"Input folder:  {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Total images:  {total_files}")
    print(f"Processing:    {len(files)}")
    if args.max_images and args.max_images < total_files:
        print(f"⚠️  SUBSET MODE: Processing {len(files)}/{total_files} images")
        print("   This is RECOMMENDED for large datasets!")
    print(f"Codec:         {args.codec}")
    print(f"CRF range:     {args.crf_min}-{args.crf_max}")
    print(f"Preset:        {args.preset}")
    print(f"Workers:       {args.workers}")
    print("=" * 60)

    # Estimate time
    time_per_image = 0.5  # Rough estimate (seconds)
    estimated_time = (len(files) * time_per_image) / 60  # minutes
    print(f"\n⏱️  Estimated time: ~{estimated_time:.0f} minutes")
    if estimated_time > 120:
        print(f"   ({estimated_time / 60:.1f} hours)")
    print()

    if len(files) > 10000:
        print("⚠️  WARNING: Large dataset detected!")
        print("   Consider using --max_images 5000 for faster processing.")
        print("   5k-10k images is usually sufficient for training.\n")
        response = input("Continue with full processing? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Prepare work items
    tasks = []
    for f in files:
        out_file = output_dir / f.name
        tasks.append((f, out_file, args.codec, args.crf_min, args.crf_max, args.preset))

    # Process (single-threaded is actually faster due to FFmpeg overhead)
    print("Applying video compression...")
    for task in tqdm(tasks, desc="Processing", unit="img"):
        process_image(task)

    print("\n" + "=" * 60)
    print("✓ Dataset creation complete!")
    print(f"✓ Video-compressed images saved to: {output_dir}")
    print(f"✓ Processed: {len(files)} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
