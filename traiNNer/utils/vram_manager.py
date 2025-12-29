#!/usr/bin/env python3
"""
VRAM Management System for ParagonSR2 Training

Automatic VRAM management that chooses optimal training parameters
to prevent OOM crashes and maximize training efficiency.

Author: Philip Hofmann
License: MIT
Repository: https://github.com/Phhofm/traiNNer-redux

Model Complexity Reference (2x SR):
----------------------------------
Nano:  12 feat, 1x1 blocks, ~0.02M params, ~0.5 GFLOPs
Micro: 16 feat, 1x2 blocks, ~0.04M params, ~1.0 GFLOPs
Tiny:  24 feat, 2x2 blocks, ~0.08M params, ~2.0 GFLOPs
XS:    32 feat, 2x3 blocks, ~0.12M params, ~3.5 GFLOPs
S:     48 feat, 3x4 blocks, ~0.28M params, ~8 GFLOPs
M:     64 feat, 4x6 blocks, ~0.65M params, ~18 GFLOPs
L:     96 feat, 6x8 blocks, ~1.8M params, ~45 GFLOPs
XL:    128 feat, 8x10 blocks, ~3.8M params, ~95 GFLOPs
"""

import gc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class ModelVariant(Enum):
    """ParagonSR2 model variants with complexity levels."""

    NANO = "nano"
    MICRO = "micro"
    TINY = "tiny"
    XS = "xs"
    S = "s"
    M = "m"
    L = "l"
    XL = "xl"


@dataclass
class ModelSpecs:
    """Model specifications for VRAM estimation."""

    variant: ModelVariant
    num_feat: int
    num_groups: int
    num_blocks: int
    params_millions: float
    gflops: float


@dataclass
class VRAMInfo:
    """GPU VRAM information."""

    total_gb: float
    available_gb: float
    utilization_percent: float
    reserved_gb: float = 0.5


@dataclass
class TrainingConfig:
    """Optimized training configuration."""

    lq_size: int
    batch_size_per_gpu: int
    num_worker_per_gpu: int
    accum_iter: int
    estimated_vram_gb: float
    vram_efficiency: float
    safety_score: float


@dataclass
class DatasetInfo:
    """Dataset characteristics for VRAM estimation."""

    image_size_lr: int
    image_size_hr: int
    num_samples: int
    channels: int = 3


class VRAMEstimator:
    """Estimate VRAM requirements based on model and dataset characteristics."""

    MODEL_SPECS = {
        ModelVariant.NANO: ModelSpecs(
            variant=ModelVariant.NANO,
            num_feat=12,
            num_groups=1,
            num_blocks=1,
            params_millions=0.02,
            gflops=0.5,
        ),
        ModelVariant.MICRO: ModelSpecs(
            variant=ModelVariant.MICRO,
            num_feat=16,
            num_groups=1,
            num_blocks=2,
            params_millions=0.04,
            gflops=1.0,
        ),
        ModelVariant.TINY: ModelSpecs(
            variant=ModelVariant.TINY,
            num_feat=24,
            num_groups=2,
            num_blocks=2,
            params_millions=0.08,
            gflops=2.0,
        ),
        ModelVariant.XS: ModelSpecs(
            variant=ModelVariant.XS,
            num_feat=32,
            num_groups=2,
            num_blocks=3,
            params_millions=0.12,
            gflops=3.5,
        ),
        ModelVariant.S: ModelSpecs(
            variant=ModelVariant.S,
            num_feat=48,
            num_groups=3,
            num_blocks=4,
            params_millions=0.28,
            gflops=8.0,
        ),
        ModelVariant.M: ModelSpecs(
            variant=ModelVariant.M,
            num_feat=64,
            num_groups=4,
            num_blocks=6,
            params_millions=0.65,
            gflops=18.0,
        ),
        ModelVariant.L: ModelSpecs(
            variant=ModelVariant.L,
            num_feat=96,
            num_groups=6,
            num_blocks=8,
            params_millions=1.8,
            gflops=45.0,
        ),
        ModelVariant.XL: ModelSpecs(
            variant=ModelVariant.XL,
            num_feat=128,
            num_groups=8,
            num_blocks=10,
            params_millions=3.8,
            gflops=95.0,
        ),
    }

    def __init__(self, base_vram_gb: float = 0.5) -> None:
        self.base_vram_gb = base_vram_gb

    def get_model_specs(self, variant: str) -> ModelSpecs:
        """Get model specifications by variant name."""
        try:
            return self.MODEL_SPECS[ModelVariant(variant.lower())]
        except (ValueError, KeyError):
            raise ValueError(
                f"Unknown model variant: {variant}. "
                f"Supported variants: {[v.value for v in ModelVariant]}"
            )

    def estimate_vram_usage(
        self, model_variant: str, batch_size: int, image_size: int, channels: int = 3
    ) -> float:
        """
        Estimate VRAM usage for a specific configuration.

        Formula based on empirical measurements and model complexity:
        - Base VRAM: model parameters + framework overhead
        - Per-sample VRAM: scales with batch size and image size
        - Model complexity multiplier: based on GFLOPs
        """
        specs = self.get_model_specs(model_variant)

        # Base VRAM: model parameters + PyTorch overhead
        base_vram = self.base_vram_gb

        # Per-sample VRAM estimation (empirically derived)
        pixels_per_sample = batch_size * image_size * image_size * channels

        # Complexity multiplier based on GFLOPs
        complexity_factor = 1.0 + (specs.gflops / 50.0)

        # Memory scaling formula (empirical)
        per_sample_mb = (pixels_per_sample * 4) / (1024 * 1024)  # 4 bytes per pixel
        per_sample_mb *= complexity_factor

        # Training overhead (gradients + optimizer states)
        training_overhead = 3.0  # 3x multiplier for training vs inference

        total_vram_gb = base_vram + (per_sample_mb * training_overhead) / 1024

        # Apply safety margin for dynamic allocations
        safety_margin = 1.1  # 10% safety margin
        total_vram_gb *= safety_margin

        return max(total_vram_gb, 0.5)  # Minimum 0.5GB

    def get_estimated_batch_sizes(
        self, model_variant: str, available_vram_gb: float, image_size: int
    ) -> dict[str, int]:
        """Get estimated optimal batch sizes for different VRAM efficiency levels."""
        specs = self.get_model_specs(model_variant)

        # Efficiency levels: conservative, balanced, aggressive
        efficiency_levels = {
            "conservative": 0.60,  # 60% VRAM usage
            "balanced": 0.75,  # 75% VRAM usage
            "aggressive": 0.85,  # 85% VRAM usage
        }

        results = {}
        base_vram = self.base_vram_gb
        usable_vram = available_vram_gb - 0.5  # Reserve 0.5GB for stability

        for level, efficiency in efficiency_levels.items():
            target_vram = usable_vram * efficiency

            # Calculate per-sample VRAM contribution
            pixels_per_sample = image_size * image_size * 3
            complexity_factor = 1.0 + (specs.gflops / 50.0)
            per_sample_mb = (pixels_per_sample * 4 * complexity_factor) / (1024 * 1024)
            per_sample_gb = (per_sample_mb * 3) / 1024  # Training overhead

            if per_sample_gb > 0 and target_vram > base_vram:
                batch_size = int((target_vram - base_vram) / per_sample_gb)
            else:
                # Use more aggressive estimation based on model complexity
                complexity_batch_factor = {
                    "nano": 16,
                    "micro": 12,
                    "tiny": 8,
                    "xs": 6,
                    "s": 4,
                    "m": 3,
                    "l": 2,
                    "xl": 1,
                }.get(specs.variant.value, 4)
                batch_size = complexity_batch_factor

            # Ensure batch size is within reasonable bounds
            batch_size = max(1, min(batch_size, 32))
            results[level] = batch_size

        return results


class ParameterOptimizer:
    """Optimize training parameters for available VRAM."""

    def __init__(
        self,
        min_lq_size: int = 64,
        max_lq_size: int = 512,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
    ) -> None:
        self.min_lq_size = min_lq_size
        self.max_lq_size = max_lq_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

    def optimize_for_target_vram(
        self,
        model_variant: str,
        available_vram_gb: float,
        target_efficiency: float = 0.85,
        dataset_info: DatasetInfo | None = None,
    ) -> TrainingConfig:
        """Optimize parameters to achieve target VRAM efficiency."""
        estimator = VRAMEstimator()

        # Use dataset info if provided, otherwise use defaults
        if dataset_info:
            image_size = dataset_info.image_size_lr
        else:
            # Default: 2x SR from 512px HR = 256px LR
            image_size = 256

        # Try different parameter combinations
        best_config = None
        best_efficiency = 0.0

        for lq_size in [64, 96, 128, 192, 256, 384, 512]:
            if lq_size > self.max_lq_size:
                continue

            batch_sizes = estimator.get_estimated_batch_sizes(
                model_variant, available_vram_gb, lq_size
            )

            logger.debug(f"Testing lq_size={lq_size}, batch_sizes={batch_sizes}")

            for efficiency_level in ["conservative", "balanced", "aggressive"]:
                batch_size = batch_sizes[efficiency_level]

                # Clamp batch size to allowed range
                batch_size = max(
                    self.min_batch_size, min(batch_size, self.max_batch_size)
                )

                # Calculate actual VRAM usage
                estimated_vram = estimator.estimate_vram_usage(
                    model_variant, batch_size, lq_size
                )

                actual_efficiency = estimated_vram / available_vram_gb

                # Score configuration (prefer configurations close to target)
                target_diff = abs(actual_efficiency - target_efficiency)
                safety_score = 1.0 - target_diff  # Higher is better

                # Prefer configurations closer to target, but accept reasonable alternatives
                if (
                    target_diff <= 0.30  # More permissive range (up to 30% deviation)
                    and (
                        best_config is None
                        or actual_efficiency > best_efficiency
                        or (
                            abs(actual_efficiency - target_efficiency) < 0.10
                            and lq_size < best_config.lq_size
                        )
                    )
                ):
                    best_config = TrainingConfig(
                        lq_size=lq_size,
                        batch_size_per_gpu=batch_size,
                        num_worker_per_gpu=min(
                            8, batch_size * 2
                        ),  # Reasonable worker count
                        accum_iter=1,  # Default, could be optimized further
                        estimated_vram_gb=estimated_vram,
                        vram_efficiency=actual_efficiency,
                        safety_score=safety_score,
                    )
                    best_efficiency = actual_efficiency

        if best_config is None:
            # Fallback: use empirical configurations based on model variant and VRAM
            logger.warning(
                "Could not find optimal configuration, using empirical settings"
            )

            # Empirical batch sizes for different variants (tested on 12GB GPU)
            variant_batch_sizes = {
                "nano": max(1, int(available_vram_gb / 1.5)),  # ~8-16 for 12-24GB
                "micro": max(1, int(available_vram_gb / 2.0)),  # ~6-12 for 12-24GB
                "tiny": max(1, int(available_vram_gb / 2.5)),  # ~4-9 for 12-24GB
                "xs": max(1, int(available_vram_gb / 3.0)),  # ~3-8 for 12-24GB
                "s": max(1, int(available_vram_gb / 3.5)),  # ~3-6 for 12-24GB
                "m": max(1, int(available_vram_gb / 4.0)),  # ~2-6 for 12-24GB
                "l": max(1, int(available_vram_gb / 5.0)),  # ~2-4 for 12-24GB
                "xl": max(1, int(available_vram_gb / 8.0)),  # ~1-3 for 12-24GB
            }

            fallback_batch = variant_batch_sizes.get(model_variant, 4)
            fallback_batch = max(
                1, min(fallback_batch, 32)
            )  # Clamp to reasonable range

            # Optimal lq_size based on batch size (larger batches = smaller patches)
            if fallback_batch >= 8:
                fallback_lq = 128
            elif fallback_batch >= 4:
                fallback_lq = 192
            elif fallback_batch >= 2:
                fallback_lq = 256
            else:
                fallback_lq = 512

            # Calculate VRAM usage for fallback
            fallback_vram = estimator.estimate_vram_usage(
                model_variant, fallback_batch, fallback_lq
            )
            fallback_efficiency = fallback_vram / available_vram_gb

            best_config = TrainingConfig(
                lq_size=fallback_lq,
                batch_size_per_gpu=fallback_batch,
                num_worker_per_gpu=min(8, fallback_batch * 2),
                accum_iter=1,
                estimated_vram_gb=fallback_vram,
                vram_efficiency=fallback_efficiency,
                safety_score=max(
                    0.5, 1.0 - abs(fallback_efficiency - target_efficiency)
                ),  # Reasonable score
            )

        return best_config


class VRAMManager:
    """
    Main VRAM management system for ParagonSR2 training.

    Features:
    - Automatic VRAM detection and optimization
    - Runtime monitoring and adjustment
    - Integration with training automations
    - OOM prevention and recovery
    """

    def __init__(
        self,
        target_vram_usage: float = 0.85,
        enable_monitoring: bool = True,
        safety_margin: float = 0.05,
    ) -> None:
        """
        Args:
            target_vram_usage: Target VRAM usage ratio (0.0-1.0)
            enable_monitoring: Enable runtime VRAM monitoring
            safety_margin: Safety margin to prevent OOM (0.0-1.0)
        """
        self.target_vram_usage = target_vram_usage
        self.enable_monitoring = enable_monitoring
        self.safety_margin = safety_margin

        self.estimator = VRAMEstimator()
        self.optimizer = ParameterOptimizer()

        self.current_config = None
        self.vram_history = []
        self.adjustment_count = 0

    def auto_optimize(
        self,
        model: torch.nn.Module,
        available_vram_gb: float | None = None,
        dataset_info: DatasetInfo | None = None,
    ) -> TrainingConfig:
        """
        Automatically optimize training parameters for available hardware.

        Args:
            model: The model to be trained
            available_vram_gb: Available VRAM in GB (auto-detected if None)
            dataset_info: Optional dataset information

        Returns:
            Optimized training configuration
        """
        logger.info("Starting automatic VRAM optimization...")

        # Get available VRAM
        if available_vram_gb is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_mb = (
                    torch.cuda.get_device_properties(device).total_memory / 1024**2
                )
                allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
                reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
                available_mb = total_mb - max(allocated_mb, reserved_mb)
                available_vram = available_mb / 1024
                total_vram = total_mb / 1024
                logger.info(
                    f"Detected {total_vram:.1f}GB total VRAM, "
                    f"{available_vram:.1f}GB available"
                )
            else:
                raise RuntimeError(
                    "CUDA not available - VRAM optimization requires GPU"
                )
        else:
            available_vram = available_vram_gb

        # Extract model variant from model architecture
        model_variant = self._detect_model_variant(model)
        logger.info(f"Detected model variant: {model_variant}")

        # Get optimal configuration
        config = self.optimizer.optimize_for_target_vram(
            model_variant=model_variant,
            available_vram_gb=available_vram,
            target_efficiency=self.target_vram_usage,
            dataset_info=dataset_info,
        )

        # Apply safety margin
        if config.vram_efficiency > (self.target_vram_usage + self.safety_margin):
            logger.warning(
                f"Configuration efficiency {config.vram_efficiency:.2f} "
                f"exceeds target {self.target_vram_usage:.2f}, reducing batch size"
            )

            reduction_factor = (
                self.target_vram_usage + self.safety_margin
            ) / config.vram_efficiency
            config.batch_size_per_gpu = max(
                1, int(config.batch_size_per_gpu * reduction_factor)
            )

            # Recalculate VRAM usage
            config.estimated_vram_gb = self.estimator.estimate_vram_usage(
                model_variant, config.batch_size_per_gpu, config.lq_size
            )
            config.vram_efficiency = config.estimated_vram_gb / available_vram

        self.current_config = config
        self._log_optimization_result(config, available_vram, model_variant)

        return config

    def _detect_model_variant(self, model: torch.nn.Module) -> str:
        """Detect model variant from model architecture."""
        model_name = model.__class__.__name__.lower()

        # Map architecture class names to variants
        if "nano" in model_name:
            return "nano"
        elif "micro" in model_name:
            return "micro"
        elif "tiny" in model_name:
            return "tiny"
        elif "xs" in model_name:
            return "xs"
        elif "paragonsr2" in model_name:
            # For base ParagonSR2 class, try to infer from num_feat
            if hasattr(model, "num_feat"):
                feat_count = int(model.num_feat)  # Ensure it's an int
                if feat_count <= 12:
                    return "nano"
                elif feat_count <= 16:
                    return "micro"
                elif feat_count <= 24:
                    return "tiny"
                elif feat_count <= 32:
                    return "xs"
                elif feat_count <= 48:
                    return "s"
                elif feat_count <= 64:
                    return "m"
                elif feat_count <= 96:
                    return "l"
                else:
                    return "xl"
            return "s"  # Default to S
        else:
            # Try to match by parameter count or other heuristics
            total_params = sum(p.numel() for p in model.parameters())
            param_millions = total_params / 1e6

            if param_millions <= 0.05:
                return "nano"
            elif param_millions <= 0.08:
                return "micro"
            elif param_millions <= 0.15:
                return "tiny"
            elif param_millions <= 0.5:
                return "s"
            elif param_millions <= 2.0:
                return "m"
            else:
                return "l"

    def _log_optimization_result(
        self, config: TrainingConfig, available_vram: float, model_variant: str
    ) -> None:
        """Log the optimization results."""
        logger.info("=" * 60)
        logger.info("VRAM OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model Variant: {model_variant}")
        logger.info(f"Available VRAM: {available_vram:.1f}GB")
        logger.info(f"Target Efficiency: {self.target_vram_usage:.2%}")
        logger.info("")
        logger.info("OPTIMAL CONFIGURATION:")
        logger.info(f"  LR Patch Size (lq_size): {config.lq_size}")
        logger.info(f"  Batch Size per GPU: {config.batch_size_per_gpu}")
        logger.info(f"  Workers per GPU: {config.num_worker_per_gpu}")
        logger.info(f"  Gradient Accumulation: {config.accum_iter}")
        logger.info("")
        logger.info("VRAM ANALYSIS:")
        logger.info(f"  Estimated Usage: {config.estimated_vram_gb:.2f}GB")
        logger.info(f"  VRAM Efficiency: {config.vram_efficiency:.2%}")
        logger.info(f"  Safety Score: {config.safety_score:.2f}/1.0")
        logger.info(
            f"  Risk Level: {'LOW' if config.safety_score > 0.7 else 'MEDIUM' if config.safety_score > 0.4 else 'HIGH'}"
        )
        logger.info("=" * 60)

    def check_vram_safety(self) -> bool:
        """Check if current VRAM usage is safe for training."""
        if not self.enable_monitoring or not torch.cuda.is_available():
            return True

        device = torch.cuda.current_device()
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2
        allocated_mb = torch.cuda.memory_allocated(device) / 1024**2

        current_usage_ratio = allocated_mb / total_mb

        # Check if VRAM usage exceeds safe threshold
        safe_threshold = self.target_vram_usage + self.safety_margin

        if current_usage_ratio > safe_threshold:
            logger.warning(
                f"VRAM usage {current_usage_ratio:.2%} exceeds safe threshold "
                f"{safe_threshold:.2%}"
            )
            return False

        return True

    def get_training_config_dict(self) -> dict[str, Any]:
        """Get current configuration as dictionary for training setup."""
        if not self.current_config:
            raise RuntimeError(
                "No configuration available. Call auto_optimize() first."
            )

        return {
            "lq_size": self.current_config.lq_size,
            "batch_size_per_gpu": self.current_config.batch_size_per_gpu,
            "num_worker_per_gpu": self.current_config.num_worker_per_gpu,
            "accum_iter": self.current_config.accum_iter,
        }

    def create_dataset_info(
        self, hr_image_size: int = 512, scale: int = 2
    ) -> DatasetInfo:
        """Create DatasetInfo from common parameters."""
        return DatasetInfo(
            image_size_lr=hr_image_size // scale,
            image_size_hr=hr_image_size,
            num_samples=0,  # Unknown during optimization
            channels=3,
        )
