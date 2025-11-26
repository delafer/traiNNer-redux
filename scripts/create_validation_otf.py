#!/usr/bin/env python3
"""
Create OTF-degraded validation dataset for ParagonSR training.

Pre-applies ParagonSR OTF degradations to validation images using the same
settings as training, enabling consistent evaluation during training.

Usage:
    python create_validation_otf.py \\
        --input datasets/val/hr \\
        --output datasets/val/lr_youtube \\
        --sequence youtube \\
        --count 100 \\
        --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).parents[2]
sys.path.insert(0, str(repo_root))

from traiNNer.models.paragon_comprehensive_sequences import create_all_sequences


def preprocess_image(img_path: Path) -> torch.Tensor:
    """Load image as torch tensor."""
    img = Image.open(img_path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
    return img


def postprocess_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for saving."""
    img = img_tensor.squeeze(0).clamp(0, 1).cpu().numpy()
    img = (img * 255).round().astype(np.uint8).transpose(1, 2, 0)  # CHW -> HWC
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create OTF-degraded validation dataset"
    )
    parser.add_argument("--input", required=True, help="Input HR validation directory")
    parser.add_argument("--output", required=True, help="Output LR directory")
    parser.add_argument(
        "--sequence",
        default="youtube",
        choices=["youtube", "tiktok", "streaming", "dvdrip", "social_multi", "all"],
        help="Degradation sequence to apply",
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of images to process"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--scale", type=int, default=2, help="SR scale factor")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input images
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_exts])[
        : args.count
    ]

    if not images:
        print(f"Error: No images found in {input_dir}")
        return

    print(f"Found {len(images)} images, processing {min(len(images), args.count)}...")

    # Get sequences
    from traiNNer.models.paragon_sequences import SequenceController

    all_sequences = create_all_sequences(include_video=True)

    # Filter to specific sequence if requested
    if args.sequence != "all":
        all_sequences = [s for s in all_sequences if args.sequence in s.name.lower()]

    if not all_sequences:
        print(f"Error: No sequence found matching '{args.sequence}'")
        return

    controller = SequenceController(all_sequences)

    # Create mock opt for degradations
    class MockOpt:
        pass

    opt = MockOpt()
    opt.scale = args.scale

    # Process images
    metadata = {"sequence": args.sequence, "seed": args.seed, "images": []}

    for img_path in tqdm(images, desc="Degrading images"):
        try:
            # Load image
            img_tensor = (
                preprocess_image(img_path).cuda()
                if torch.cuda.is_available()
                else preprocess_image(img_path)
            )

            # Apply degradation sequence
            degraded = controller.apply_sequence(img_tensor, opt)

            # Downscale to LR resolution
            h, w = degraded.shape[2:]
            degraded = torch.nn.functional.interpolate(
                degraded,
                size=(h // args.scale, w // args.scale),
                mode="bicubic",
                align_corners=False,
            )

            # Save
            output_path = output_dir / img_path.name
            img_array = postprocess_image(degraded)
            cv2.imwrite(str(output_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            metadata["images"].append(
                {"input": str(img_path.name), "output": str(output_path.name)}
            )

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # Save metadata
    metadata_path = output_dir / "degradation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Successfully degraded {len(metadata['images'])} images")
    print(f"✓ Output: {output_dir}")
    print(f"✓ Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
