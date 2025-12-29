#!/usr/bin/env python3
"""
Quick ParagonSR Toy Model Trainer
Author: Philip Hofmann

Description:
Creates quick toy models for benchmarking (100 iterations, very fast training).
Only designed for speed benchmarking, not quality.

Usage:
python3 scripts/benchmarking/train_toy_models.py --output_dir /path/to/output --variant s --scale 4 --iterations 100
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traiNNer.archs.paragonsr_arch import (
    paragonsr_l,
    paragonsr_m,
    paragonsr_s,
    paragonsr_tiny,
    paragonsr_xl,
    paragonsr_xs,
)


class SimpleToYDataset(Dataset):
    """Simple toy dataset for quick training."""

    def __init__(self, size: int = 256, num_samples: int = 100) -> None:
        self.size = size
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random noise image
        lr_img = torch.randn(3, self.size // 4, self.size // 4)  # Low-res input
        hr_img = torch.randn(3, self.size, self.size)  # High-res target

        # Normalize to [0, 1] to avoid numerical issues
        lr_img = torch.clamp(lr_img, 0, 1)
        hr_img = torch.clamp(hr_img, 0, 1)

        return lr_img, hr_img


def get_model_func(variant: str):
    """Get model function for variant."""
    variants = {
        "tiny": paragonsr_tiny,
        "xs": paragonsr_xs,
        "s": paragonsr_s,
        "m": paragonsr_m,
        "l": paragonsr_l,
        "xl": paragonsr_xl,
    }

    if variant not in variants:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from: {list(variants.keys())}"
        )

    return variants[variant]


def train_toy_model(variant: str, scale: int, output_dir: str, iterations: int = 100):
    """Train a toy model for benchmarking."""
    print(f"🚀 Training ParagonSR-{variant.upper()} {scale}x Toy Model")
    print(f"   Iterations: {iterations}")
    print(f"   Output: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get model
    model_func = get_model_func(variant)
    model = model_func(scale=scale)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Create toy dataset
    dataset = SimpleToyDataset(size=256, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"   Device: {device}")
    print(f"   Dataset: {len(dataset)} samples")
    print("   Starting training...")

    # Training loop
    model.train()
    for epoch in range(iterations):
        total_loss = 0
        num_batches = 0

        for batch_idx, (lr_img, hr_img) in enumerate(dataloader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(lr_img)

            # Loss (simple MSE between output and target)
            loss = criterion(output, hr_img)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress every 20 iterations
            if (epoch + 1) % 20 == 0 and batch_idx == 0:
                avg_loss = total_loss / num_batches
                print(f"   Epoch {epoch + 1}/{iterations}, Loss: {avg_loss:.4f}")
                total_loss = 0
                num_batches = 0

        # Save checkpoint every 50 iterations
        if (epoch + 1) % 50 == 0:
            checkpoint_path = output_path / f"{variant}_{scale}x_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                checkpoint_path,
            )
            print(f"   💾 Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_path / f"{variant}_{scale}x_toy_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"   ✅ Final model saved: {final_path}")

    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   📊 Model parameters: {param_count:,}")

    return str(final_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ParagonSR Toy Models for Benchmarking"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["tiny", "xs", "s", "m", "l", "xl"],
        help="Model variant to train",
    )
    parser.add_argument(
        "--scale", type=int, required=True, choices=[2, 4, 8], help="Scale factor"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for trained models"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Training iterations (default: 100)"
    )

    args = parser.parse_args()

    # Train toy model
    model_path = train_toy_model(
        args.variant, args.scale, args.output_dir, args.iterations
    )

    print("\n🎉 Toy model training completed!")
    print(f"📁 Model saved to: {model_path}")
    print("💡 Next steps:")
    print("   1. Use paragon_deploy.py to fuse and export to ONNX")
    print("   2. Use benchmark_paragon.py to measure performance")


if __name__ == "__main__":
    main()
