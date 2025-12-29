#!/usr/bin/env python3
"""
High-Speed Tiling & Pre-Filtering Pipeline (Stage 1)
Author: Philip Hofmann

Description:
This script serves as the first stage in a three-stage data curation pipeline
designed to create high-quality ground truth datasets for Super-Resolution
(SISR) model training.

Its primary purpose is to act as a high-speed "sledgehammer," processing raw,
large-scale image datasets with maximum efficiency. It performs multi-scale
tiling and uses an arsenal of extremely fast CPU and GPU metrics to aggressively
discard tiles with undeniable technical flaws such as blur, noise, compression
artifacts, low information, and aliasing.

The output is a large but pre-cleaned set of 512x512 "candidate" tiles and a
detailed CSV log, which serve as the input for the more specialized Stage 2.

Key Features:
- Multi-scale processing (100%, 75%, 50%, 25%, adaptive)
- Fast CPU metrics for initial filtering
- BRISQUE GPU model for quality assessment
- Comprehensive logging and error handling
- Memory-efficient batch processing

Example Usage:
    python stage1_prefilter.py input_folder output_folder output.csv --prefix "my_dataset"

Requirements:
- Python 3.7+
- PyTorch, PyIQa, OpenCV, PIL
- CUDA-capable GPU (recommended)

Author: Philip Hofmann
License: CC0 (Public Domain)
Version: 1.0.0
"""

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

# ============================= CONFIGURATION =============================
TILE_SIZE = 512
# This batch size should be tuned based on hardware for maximum throughput.
# A value of 64-96 is a good starting point for a 12GB prosumer GPU.
BATCH_SIZE = 64

# Default thresholds are set to be permissive but effective. They can be
# overridden via command-line arguments.
DEFAULT_THRESHOLDS = {
    "entropy": 5.0,
    "oversharpen": 3000.0,
    "blockiness": 40.0,
    "brisque": 40.0,
}

# These bounds are fundamental to the pre-filtering logic and are hard-coded
# to ensure a consistent baseline for rejecting the worst technical flaws.
CPU_FILTER_BOUNDS = {
    "oversharpen_lower": 50.0,  # Rejects blur, fog, and heavy out-of-focus areas
    "aliasing_upper": 0.35,  # Rejects moiré patterns and downscaling artifacts
    "contrast_lower": 15.0,  # Rejects washed-out, hazy, low-contrast images
}

# Default scales to process.
SCALES_CONFIG = [(1.0, "100"), (0.75, "75"), (0.5, "50"), (0.25, "25")]


# ======================= FAST QUALITY/ARTIFACT METRICS ======================
def compute_entropy(tile_gray):
    """Computes the Shannon entropy, a measure of information density."""
    hist = cv2.calcHist([tile_gray], [0], None, [256], [0, 256])
    hist /= hist.sum() + 1e-9
    return -np.sum(hist * np.log2(hist + 1e-9))


def compute_blockiness(tile_gray):
    """Computes a simple metric for JPEG block compression artifacts."""
    h, w = tile_gray.shape
    if h < 16 or w < 16:
        return 0.0
    h_diff = sum(
        np.sum(np.abs(tile_gray[i, :] - tile_gray[i - 1, :])) for i in range(8, h, 8)
    )
    v_diff = sum(
        np.sum(np.abs(tile_gray[:, j] - tile_gray[:, j - 1])) for j in range(8, w, 8)
    )
    return (h_diff + v_diff) / (h * w)


def compute_aliasing_score(tile_gray):
    """Computes a score based on high-frequency energy to detect aliasing/moiré."""
    f_transform = np.fft.fft2(tile_gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    rows, cols = tile_gray.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    radius = int(min(center_row, center_col) * 0.75)
    cv2.circle(mask, (center_col, center_row), radius, 0, -1)
    high_freq_energy = np.abs(f_transform_shifted[mask == 1]).mean()
    total_energy = np.abs(f_transform_shifted).mean()
    return high_freq_energy / (total_energy + 1e-9)


def compute_contrast(tile_gray):
    """Calculates contrast as the standard deviation of pixel intensities."""
    return tile_gray.std()


# ======================= PROCESSING WORKER ======================
worker_models = {}


def init_worker(brisque_sd, device_str) -> None:
    """Initializes the lightweight BRISQUE model for each worker process."""
    pid = os.getpid()
    device = torch.device(device_str)
    try:
        brisque_metric = pyiqa.create_metric("brisque", device=device)
        brisque_metric.load_state_dict(brisque_sd, strict=False)
        brisque_metric.eval()
        worker_models["brisque"] = brisque_metric
    except Exception as e:
        print(
            f"FATAL: Worker (PID: {pid}) failed to initialize BRISQUE model. Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def process_image_task(args_tuple):
    """The core task for processing a single image, executed by a worker."""
    (image_path, output_folder, prefix, scales, thresholds, device_str) = args_tuple

    device = torch.device(device_str)

    def process_and_tile(scaled_img, suffix):
        """Tiles an image, runs all filters, and saves passing tiles and scores."""
        h, w, _ = scaled_img.shape
        tiles_y, tiles_x = h // TILE_SIZE, w // TILE_SIZE
        if tiles_y == 0 or tiles_x == 0:
            return 0, []

        local_passed_count, local_csv_rows = 0, []

        # --- Stage 1A: Aggressive CPU Pre-filtering ---
        gpu_candidate_tiles = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                tile_f32 = scaled_img[
                    y * TILE_SIZE : (y + 1) * TILE_SIZE,
                    x * TILE_SIZE : (x + 1) * TILE_SIZE,
                    :,
                ]
                tile_gray = cv2.cvtColor(
                    (tile_f32 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                )

                # --- Arsenal of Fast CPU Checks ---
                entropy = compute_entropy(tile_gray)
                if entropy < thresholds["entropy"]:
                    continue

                contrast = compute_contrast(tile_gray)
                if contrast < CPU_FILTER_BOUNDS["contrast_lower"]:
                    continue

                oversharpen = cv2.Laplacian(tile_gray, cv2.CV_64F).var()
                if (
                    oversharpen > thresholds["oversharpen"]
                    or oversharpen < CPU_FILTER_BOUNDS["oversharpen_lower"]
                ):
                    continue

                blockiness = compute_blockiness(tile_gray)
                if blockiness > thresholds["blockiness"]:
                    continue

                aliasing = compute_aliasing_score(tile_gray)
                if aliasing > CPU_FILTER_BOUNDS["aliasing_upper"]:
                    continue

                fast_scores = {
                    "entropy": entropy,
                    "oversharpen": oversharpen,
                    "blockiness": blockiness,
                    "aliasing": aliasing,
                    "contrast": contrast,
                }
                gpu_candidate_tiles.append(
                    {"tile_f32": tile_f32, "coords": (x, y), "fast_scores": fast_scores}
                )

        # --- Stage 1B: Fast GPU Gatekeeper (BRISQUE) ---
        if not gpu_candidate_tiles:
            return 0, []

        brisque_metric = worker_models.get("brisque")
        if not brisque_metric:
            return 0, []

        for i in range(0, len(gpu_candidate_tiles), BATCH_SIZE):
            batch_data = gpu_candidate_tiles[i : i + BATCH_SIZE]
            tiles_f32_list = [item["tile_f32"] for item in batch_data]
            batch_tensor = (
                torch.tensor(np.array(tiles_f32_list)).permute(0, 3, 1, 2).to(device)
            )
            with torch.no_grad():
                brisque_scores = brisque_metric(batch_tensor)

            for j, item in enumerate(batch_data):
                brisque_score = brisque_scores[j].item()
                if brisque_score > thresholds["brisque"]:
                    continue

                local_passed_count += 1
                all_scores = item["fast_scores"]
                all_scores["brisque"] = brisque_score
                for k, v in all_scores.items():
                    all_scores[k] = round(v, 4)

                tile_name = f"{prefix}_{image_path.stem}_{suffix}_{item['coords'][1]}_{item['coords'][0]}.png"
                Image.fromarray((item["tile_f32"] * 255).astype(np.uint8)).save(
                    output_folder / tile_name
                )

                row = {
                    "tile_path": tile_name,
                    "original_image": image_path.name,
                    "prefix": prefix,
                    "scale": suffix,
                }
                row.update(all_scores)
                local_csv_rows.append(row)

        return local_passed_count, local_csv_rows

    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
        img_uint8 = np.array(img_rgb)
        img_f32 = img_uint8.astype(np.float32) / 255.0

        if img_f32.ndim == 2:
            img_f32 = np.stack([img_f32] * 3, axis=-1)
        if img_f32.shape[0] < TILE_SIZE or img_f32.shape[1] < TILE_SIZE:
            return f"Skipping {image_path.name}: too small", 0, []

        passed_count, csv_rows = 0, []
        for scale_factor, suffix in scales:
            new_h, new_w = (
                round(img_f32.shape[0] * scale_factor),
                round(img_f32.shape[1] * scale_factor),
            )
            if new_h < TILE_SIZE or new_w < TILE_SIZE:
                continue
            scaled_img = (
                cv2.resize(img_f32, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if scale_factor != 1.0
                else img_f32
            )
            count, rows = process_and_tile(scaled_img, suffix)
            passed_count += count
            csv_rows.extend(rows)

        min_dim = min(img_f32.shape[:2])
        if min_dim > TILE_SIZE:
            adaptive_scale = TILE_SIZE / min_dim
            new_h, new_w = (
                round(img_f32.shape[0] * adaptive_scale),
                round(img_f32.shape[1] * adaptive_scale),
            )
            scaled_img = cv2.resize(
                img_f32, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            count, rows = process_and_tile(scaled_img, "min512")
            passed_count += count
            csv_rows.extend(rows)

        return f"Processed {image_path.name}", passed_count, csv_rows
    except Exception as e:
        return f"Error processing {image_path.name}: {e}", 0, []
    finally:
        gc.collect()


def main() -> None:
    """Main function to parse arguments and orchestrate the filtering process."""
    parser = argparse.ArgumentParser(
        description="Stage 1: High-Speed Tiling and Pre-Filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_folder", help="Folder with raw input images.")
    parser.add_argument(
        "output_folder", help="Output folder for Stage 1 candidate tiles."
    )
    parser.add_argument("csv_file", help="Output CSV log for Stage 1 results.")
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="A unique prefix for this dataset run (e.g., 'unsplash').",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[s[0] for s in SCALES_CONFIG],
        help="Scale factors to process.",
    )
    parser.add_argument(
        "--entropy_th", type=float, default=DEFAULT_THRESHOLDS["entropy"]
    )
    parser.add_argument(
        "--oversharpen_th", type=float, default=DEFAULT_THRESHOLDS["oversharpen"]
    )
    parser.add_argument(
        "--blockiness_th", type=float, default=DEFAULT_THRESHOLDS["blockiness"]
    )
    parser.add_argument(
        "--brisque_th", type=float, default=DEFAULT_THRESHOLDS["brisque"]
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=int(os.cpu_count() / 1.5) // 2 or 1,
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Main process started, using device: {device_str}")

    print("Preparing IQA models for workers (BRISQUE only)...")
    try:
        brisque_metric_cpu = pyiqa.create_metric("brisque", device="cpu")
        brisque_sd = brisque_metric_cpu.state_dict()
        del brisque_metric_cpu
    except Exception as e:
        print(f"FATAL: Could not load BRISQUE model. Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    scales_config = [(scale, str(int(scale * 100))) for scale in args.scales]
    image_paths = sorted(
        [
            p
            for p in Path(args.input_folder).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ]
    )
    thresholds = {k.replace("_th", ""): v for k, v in vars(args).items() if "_th" in k}
    tasks = [
        (path, output_path, args.prefix, scales_config, thresholds, device_str)
        for path in image_paths
    ]

    total_passed = 0
    fieldnames = [
        "tile_path",
        "original_image",
        "prefix",
        "scale",
        "entropy",
        "oversharpen",
        "blockiness",
        "aliasing",
        "contrast",
        "brisque",
    ]

    with open(args.csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        init_args = (brisque_sd, device_str)
        with mp.Pool(
            processes=args.max_workers, initializer=init_worker, initargs=init_args
        ) as pool:
            pbar = tqdm(
                pool.imap_unordered(process_image_task, tasks),
                total=len(tasks),
                desc="Stage 1 Filtering",
            )
            for message, count, rows in pbar:
                if "Error" in message:
                    print(message, file=sys.stderr)

                if rows:
                    total_passed += count
                    writer.writerows(rows)

    print(f"\nStage 1 complete! Logged and saved {total_passed} candidate tiles.")


if __name__ == "__main__":
    # 'spawn' is the most robust start method for multiprocessing, especially with CUDA.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # It can only be set once per program.
    main()
