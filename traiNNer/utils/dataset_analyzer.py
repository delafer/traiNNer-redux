#!/usr/bin/env python3
"""
Dataset Analysis Module for Intelligent Auto-Calibration
Analyzes training data to automatically determine optimal dynamic loss scheduling parameters.

This module provides automatic dataset complexity analysis including:
- Texture variance analysis (complexity of textures and patterns)
- Edge density detection (amount of detail and sharp transitions)
- Color variation measurement (diversity of colors and lighting conditions)

Author: Philip Hofmann
"""

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class DatasetAnalyzer:
    """
    Analyzes training dataset to determine complexity characteristics for
    intelligent dynamic loss scheduling parameter optimization.

    Features:
    - Texture variance analysis using local variance filters
    - Edge density detection using gradient magnitude
    - Color variation measurement using color histogram analysis
    - Automatic complexity scoring for parameter optimization
    - Batch processing for efficient analysis
    """

    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize the dataset analyzer.

        Args:
            device: Device to use for analysis (cuda/cpu)
        """
        self.device = device

    def analyze_batch(self, batch: Tensor) -> dict[str, float]:
        """
        Analyze a batch of training images to determine complexity metrics.

        Args:
            batch: Batch of images [B, C, H, W] in range [0, 1]

        Returns:
            Dictionary containing complexity metrics:
            - texture_variance: 0.0-1.0, higher = more complex textures
            - edge_density: 0.0-1.0, higher = more edges/details
            - color_variation: 0.0-1.0, higher = more color diversity
            - overall_complexity: 0.0-1.0, combined complexity score
        """
        if batch.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {batch.dim()}D")

        # Ensure batch is on correct device
        batch = batch.to(self.device)

        # Analyze each metric
        texture_variance = self._analyze_texture_variance(batch)
        edge_density = self._analyze_edge_density(batch)
        color_variation = self._analyze_color_variance(batch)

        # Combine into overall complexity score
        overall_complexity = (texture_variance + edge_density + color_variation) / 3.0

        return {
            "texture_variance": texture_variance,
            "edge_density": edge_density,
            "color_variation": color_variation,
            "overall_complexity": overall_complexity,
        }

    def _analyze_texture_variance(self, batch: Tensor) -> float:
        """
        Analyze texture complexity using local variance.

        Uses a Sobel filter to detect local variations, then computes
        variance statistics to measure texture complexity.
        """
        # Convert to grayscale for texture analysis
        if batch.size(1) == 3:
            # RGB to grayscale conversion
            grayscale = (
                0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
            )
        else:
            grayscale = batch

        # Apply Sobel filters for edge detection
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=grayscale.dtype,
            device=self.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=grayscale.dtype,
            device=self.device,
        )

        # Reshape for convolution
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        # Apply Sobel filters
        grad_x = F.conv2d(grayscale, sobel_x, padding=1)
        grad_y = F.conv2d(grayscale, sobel_y, padding=1)

        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Compute local variance in 3x3 windows
        kernel = torch.ones(1, 1, 3, 3, device=self.device) / 9.0
        local_mean = F.conv2d(gradient_magnitude, kernel, padding=1)
        local_variance = (
            F.conv2d(gradient_magnitude**2, kernel, padding=1) - local_mean**2
        )

        # Compute variance statistics
        variance_mean = torch.mean(local_variance)
        variance_std = torch.std(local_variance)

        # Normalize to [0, 1] range
        # Higher variance indicates more complex textures
        normalized_variance = torch.clamp(
            variance_mean / (variance_std + 1e-8), 0.0, 1.0
        )

        return float(normalized_variance)

    def _analyze_edge_density(self, batch: Tensor) -> float:
        """
        Analyze edge density using gradient-based edge detection.

        Counts strong gradients to measure amount of detail and sharp transitions.
        """
        # Convert to grayscale
        if batch.size(1) == 3:
            grayscale = (
                0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
            )
        else:
            grayscale = batch

        # Apply Sobel filters
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=grayscale.dtype,
            device=self.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=grayscale.dtype,
            device=self.device,
        )

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        grad_x = F.conv2d(grayscale, sobel_x, padding=1)
        grad_y = F.conv2d(grayscale, sobel_y, padding=1)

        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Threshold to find strong edges (top 20% of gradients)
        threshold = torch.quantile(gradient_magnitude, 0.8)
        edge_mask = gradient_magnitude > threshold

        # Compute edge density
        edge_density = torch.mean(edge_mask.float())

        # Normalize and clamp
        normalized_edge_density = torch.clamp(
            edge_density / 0.2, 0.0, 1.0
        )  # Assume 20% is high density

        return float(normalized_edge_density)

    def _analyze_color_variance(self, batch: Tensor) -> float:
        """
        Analyze color variation using color histogram analysis.

        Measures diversity of colors and lighting conditions across the batch.
        """
        # Convert to [0, 255] for histogram analysis
        batch_255 = (batch * 255.0).clamp(0, 255)

        # Analyze each color channel separately
        color_variances = []

        for channel in range(min(3, batch.size(1))):  # RGB channels
            channel_data = batch_255[:, channel : channel + 1]  # Keep as [B, 1, H, W]

            # Compute histogram for each image in batch
            batch_histograms = []
            for img in channel_data:
                img_flat = img.flatten()
                hist = torch.histc(img_flat, bins=256, min=0, max=255)
                batch_histograms.append(hist)

            # Stack histograms
            histograms = torch.stack(batch_histograms, dim=0)  # [B, 256]

            # Compute histogram variance (spread of colors)
            hist_mean = torch.mean(histograms, dim=0)
            hist_variance = torch.var(histograms, dim=0)
            total_variance = torch.sum(hist_variance)

            # Normalize by number of bins and batch size
            normalized_variance = total_variance / (256.0 * batch.size(0))
            color_variances.append(normalized_variance)

        # Average across color channels
        avg_color_variance = torch.mean(torch.stack(color_variances))

        # Normalize to [0, 1] range
        # Higher values indicate more color variation
        normalized_color_variance = torch.clamp(
            avg_color_variance / 1000.0, 0.0, 1.0
        )  # Scale factor

        return float(normalized_color_variance)

    def get_recommended_parameters(
        self, complexity_metrics: dict[str, float]
    ) -> dict[str, Any]:
        """
        Get recommended dynamic loss scheduling parameters based on dataset complexity.

        Args:
            complexity_metrics: Dictionary from analyze_batch()

        Returns:
            Dictionary of recommended parameters for dynamic loss scheduling
        """
        texture_variance = complexity_metrics["texture_variance"]
        edge_density = complexity_metrics["edge_density"]
        color_variation = complexity_metrics["color_variation"]
        overall_complexity = complexity_metrics["overall_complexity"]

        # Base adjustments based on complexity
        adjustments = {}

        if overall_complexity > 0.7:
            # High complexity dataset - need more conservative adaptation
            adjustments["momentum_factor"] = 0.9  # More responsive
            adjustments["adaptation_factor"] = 1.2  # Faster adaptation
            adjustments["stability_factor"] = 1.5  # Higher thresholds
        elif overall_complexity < 0.3:
            # Low complexity dataset - can be more aggressive
            adjustments["momentum_factor"] = 1.1  # More stable
            adjustments["adaptation_factor"] = 0.8  # Slower adaptation
            adjustments["stability_factor"] = 0.7  # Lower thresholds
        else:
            # Medium complexity - balanced approach
            adjustments["momentum_factor"] = 1.0
            adjustments["adaptation_factor"] = 1.0
            adjustments["stability_factor"] = 1.0

        # Texture-specific adjustments
        if texture_variance > 0.6:
            adjustments["texture_sensitivity"] = (
                1.3  # More sensitive to texture changes
            )
        else:
            adjustments["texture_sensitivity"] = 0.8  # Less sensitive

        # Edge-specific adjustments
        if edge_density > 0.6:
            adjustments["edge_sensitivity"] = 1.2  # More sensitive to edge details
        else:
            adjustments["edge_sensitivity"] = 0.9  # Less sensitive

        # Color-specific adjustments
        if color_variation > 0.6:
            adjustments["color_sensitivity"] = 1.1  # More sensitive to color changes
        else:
            adjustments["color_sensitivity"] = 0.9  # Less sensitive

        return adjustments


def analyze_dataset_complexity(
    dataloader, num_samples: int = 100, device: str = "cuda"
) -> dict[str, float]:
    """
    Convenience function to analyze dataset complexity from a dataloader.

    Args:
        dataloader: PyTorch DataLoader containing training data
        num_samples: Number of samples to analyze (None for all available)
        device: Device to use for analysis

    Returns:
        Dictionary containing complexity metrics
    """
    analyzer = DatasetAnalyzer(device)

    total_metrics = {
        "texture_variance": 0.0,
        "edge_density": 0.0,
        "color_variation": 0.0,
        "overall_complexity": 0.0,
    }

    sample_count = 0

    try:
        for batch_idx, batch_data in enumerate(dataloader):
            if num_samples is not None and sample_count >= num_samples:
                break

            # Extract images from batch data
            if "lq" in batch_data:
                images = batch_data["lq"]
            elif "gt" in batch_data:
                images = batch_data["gt"]
            else:
                # Try to find image data in batch
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor) and value.dim() == 4:
                        images = value
                        break
                else:
                    continue

            # Analyze batch
            batch_metrics = analyzer.analyze_batch(images)

            # Accumulate metrics
            for key, value in batch_metrics.items():
                total_metrics[key] += value

            sample_count += 1

            if batch_idx % 10 == 0:
                print(
                    f"📊 Analyzed {sample_count} batches... complexity: {batch_metrics['overall_complexity']:.3f}"
                )

    except Exception as e:
        print(f"⚠️ Error during dataset analysis: {e}")
        print("📋 Using default complexity values")
        # Return default medium complexity values
        return {
            "texture_variance": 0.5,
            "edge_density": 0.5,
            "color_variation": 0.5,
            "overall_complexity": 0.5,
        }

    # Average the metrics
    if sample_count > 0:
        for key in total_metrics:
            total_metrics[key] /= sample_count

    print("✅ Dataset analysis complete:")
    print(f"   🎨 Texture Variance: {total_metrics['texture_variance']:.3f}")
    print(f"   🔍 Edge Density: {total_metrics['edge_density']:.3f}")
    print(f"   🌈 Color Variation: {total_metrics['color_variation']:.3f}")
    print(f"   📊 Overall Complexity: {total_metrics['overall_complexity']:.3f}")

    return total_metrics
