#!/usr/bin/env python3
"""
Aesthetic Quality Gate (Stage 3)
Author: Philip Hofmann

Description:
This script is the final stage of the data curation pipeline. It acts as a
fast "art critic," using the NIMA model to perform a final aesthetic quality
assessment on the technically sound tiles from Stage 2.

NIMA is a CNN-based model trained to predict the aesthetic appeal of an image
on a scale of 1 to 10. It is significantly faster than Transformer-based
alternatives like MANIQA, making this final stage computationally feasible.

This step ensures the final dataset is not only technically clean but also
aesthetically pleasing, which is a strong proxy for the authenticity and
naturalness required for training photorealistic SISR models.
"""

import os
import sys
import argparse
import csv
import torch
import pyiqa
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import shutil
import numpy as np

# ============================= CONFIGURATION =============================
# NIMA is a CNN, allowing for a much larger batch size than MANIQA.
# Tune based on your VRAM. 64 is a good starting point.
DEFAULT_BATCH_SIZE = 96

# NIMA scores range from 1 (bad) to 10 (excellent). A threshold of 6.5
# or 7.0 is a strong starting point for selecting aesthetically pleasing images.
DEFAULT_NIMA_THRESHOLD = 4.5

# ======================= MAIN SCRIPT LOGIC ======================
def main():
    """Main function to parse arguments and orchestrate the filtering process."""
    parser = argparse.ArgumentParser(description="Stage 3: Aesthetic Quality Gate.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_csv", help="The CSV log file from Stage 2 (ARNIQA).")
    parser.add_argument("tiles_folder", help="Folder where the Stage 2 tiles are stored.")
    parser.add_argument("output_folder", help="Final output folder for the highest-quality tiles.")
    parser.add_argument("output_csv", help="The final CSV log with NIMA scores.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_NIMA_THRESHOLD, help="NIMA score above which tiles are kept.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of tiles to process at once on the GPU.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu'). Autodetects if not specified.")
    args = parser.parse_args()

    if args.device:
        device_str = args.device
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device_str)
    print(f"Process started, using device: {device}")

    # --- 1. Initialize the NIMA Model ---
    print("Loading NIMA model...")
    try:
        model = pyiqa.create_metric('nima', device=device, as_loss=False)
        model.eval()
    except Exception as e:
        print(f"FATAL: Could not load NIMA model. Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded successfully.")

    # --- 2. Setup Input/Output Paths ---
    candidate_path = Path(args.input_csv)
    tiles_base_path = Path(args.tiles_folder)
    output_tiles_path = Path(args.output_folder)
    output_tiles_path.mkdir(parents=True, exist_ok=True)
    
    tile_paths = []
    with open(candidate_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile_paths.append(tiles_base_path / row['tile_path'])

    if not tile_paths:
        print(f"Error: No tile paths found in {candidate_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(tile_paths)} candidate tiles to process from Stage 2.")

    # --- 3. Process Tiles in Batches and Filter ---
    total_passed = 0
    fieldnames = ['tile_path', 'nima_score']

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(range(0, len(tile_paths), args.batch_size), desc="Stage 3 Scoring (NIMA)")
        for i in pbar:
            batch_paths = tile_paths[i:i + args.batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    with Image.open(path) as img: img_rgb = img.convert('RGB')
                    img_np = np.array(img_rgb).astype(np.float32) / 255.0
                    batch_images.append(img_np)
                except Exception as e:
                    print(f"Warning: Could not read {path}, skipping. Error: {e}", file=sys.stderr)
                    continue

            if not batch_images: continue

            batch_tensor = torch.tensor(np.array(batch_images)).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                try:
                    scores = model(batch_tensor)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nWARNING: CUDA out of memory. Try reducing --batch_size from {args.batch_size}.", file=sys.stderr)
                        continue
                    else: raise e

            for path, score_tensor in zip(batch_paths, scores):
                # NIMA returns (mean, std), we only need the mean score
                score = score_tensor[0].item()
                
                if score >= args.threshold:
                    total_passed += 1
                    shutil.copy2(path, output_tiles_path / path.name)
                    writer.writerow({'tile_path': path.name, 'nima_score': round(score, 4)})
    
    print(f"\nStage 3 complete! Found and saved {total_passed} final high-quality tiles.")
    print(f"Results logged to: {args.output_csv}")
    print(f"Tiles saved to: {args.output_folder}")

if __name__ == "__main__":
    main()