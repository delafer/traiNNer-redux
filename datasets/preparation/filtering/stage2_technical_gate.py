#!/usr/bin/env python3
"""
Technical Quality Gate (Stage 2)
Author: Philip Hofmann

Description:
This script serves as the second stage in the data curation pipeline. It acts
as a "lab technician," using the ARNIQA model to perform a rigorous technical
quality assessment on the candidate tiles produced by Stage 1.

ARNIQA is a CNN-based model specializing in the detection of technical flaws
like noise, compression, and blur. This stage's purpose is to filter the dataset
aggressively for technical soundness, ensuring that only clean, sharp, and
artifact-free tiles proceed to the final perceptual quality assessment.

It produces both a new, smaller folder of technically sound tiles and a new
CSV log containing their ARNIQA scores.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pyiqa
import torch
from PIL import Image
from tqdm import tqdm

# ============================= CONFIGURATION =============================
# ARNIQA is a CNN, allowing for a large batch size. Tune based on VRAM.
DEFAULT_BATCH_SIZE = 96

# The technical quality bar. A stricter value creates a higher quality but
# smaller input for the final stage. 0.60 is a strong baseline.
DEFAULT_ARNIQA_THRESHOLD = 0.60


# ======================= MAIN SCRIPT LOGIC ======================
def main():
    """Main function to parse arguments and orchestrate the filtering process."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Technical Quality Gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", help="The master CSV log file from Stage 1.")
    parser.add_argument(
        "tiles_folder", help="Folder where the Stage 1 tiles are stored."
    )
    parser.add_argument(
        "output_folder", help="Output folder for tiles that pass this stage."
    )
    parser.add_argument(
        "output_csv", help="The path for the new CSV log with ARNIQA scores."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ARNIQA_THRESHOLD,
        help="ARNIQA score above which tiles are kept.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of tiles to process at once on the GPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu'). Autodetects if not specified.",
    )
    args = parser.parse_args()

    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)
    print(f"Process started, using device: {device}")

    # --- 1. Initialize the ARNIQA Model ---
    print("Loading ARNIQA model...")
    try:
        model = pyiqa.create_metric("arniqa", device=device, as_loss=False)
        model.eval()
    except Exception as e:
        print(f"FATAL: Could not load ARNIQA model. Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded successfully.")

    # --- 2. Setup Input/Output Paths ---
    candidate_path = Path(args.input_csv)
    tiles_base_path = Path(args.tiles_folder)
    output_tiles_path = Path(args.output_folder)
    output_tiles_path.mkdir(parents=True, exist_ok=True)

    tile_paths = []
    with open(candidate_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_paths.append(tiles_base_path / row["tile_path"])

    if not tile_paths:
        print(f"Error: No tile paths found in {candidate_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tile_paths)} candidate tiles to process from Stage 1.")

    # --- 3. Process Tiles in Batches and Filter ---
    total_passed = 0
    fieldnames = ["tile_path", "arniqa_score"]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(
            range(0, len(tile_paths), args.batch_size), desc="Stage 2 Scoring (ARNIQA)"
        )
        for i in pbar:
            batch_paths = tile_paths[i : i + args.batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        img_rgb = img.convert("RGB")
                    img_np = np.array(img_rgb).astype(np.float32) / 255.0
                    batch_images.append(img_np)
                except Exception as e:
                    print(
                        f"Warning: Could not read {path}, skipping. Error: {e}",
                        file=sys.stderr,
                    )
                    continue

            if not batch_images:
                continue

            batch_tensor = (
                torch.tensor(np.array(batch_images)).permute(0, 3, 1, 2).to(device)
            )

            with torch.no_grad():
                try:
                    scores = model(batch_tensor)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(
                            f"\nWARNING: CUDA out of memory. Try reducing --batch_size from {args.batch_size}.",
                            file=sys.stderr,
                        )
                        continue
                    else:
                        raise e

            for path, score_tensor in zip(batch_paths, scores):
                score = score_tensor.item()

                if score >= args.threshold:
                    total_passed += 1
                    shutil.copy2(path, output_tiles_path / path.name)
                    writer.writerow(
                        {"tile_path": path.name, "arniqa_score": round(score, 4)}
                    )

    print(
        f"\nStage 2 complete! Found and saved {total_passed} technically sound tiles."
    )
    print(f"Results logged to: {args.output_csv}")
    print(f"Tiles saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
