"""
Enhanced Real-ESRGAN + ParagonSR Combined Degradation Sequences.

This module provides comprehensive sequences that combine both Real-ESRGAN
traditional degradations and ParagonSR extended degradations to simulate
real-world photography and internet workflows.

Author: Philip Hofmann
"""

from traiNNer.models.paragon_sequences import (
    DegradationSequence,
    DegradationStep,
    SequenceController,
)
from traiNNer.utils import RNG


def create_comprehensive_sequences() -> list[DegradationSequence]:
    """Create comprehensive sequences combining Real-ESRGAN + ParagonSR degradations."""

    # =========================================================
    # REALISTIC PHOTOGRAPHY WORKFLOW SEQUENCES
    # =========================================================

    # Professional Photography to Internet Workflow
    professional_to_internet = DegradationSequence(
        name="professional_to_internet",
        probability=0.2,
        repeat=1,
        repeat_probability=0.2,  # Sometimes goes through multiple platforms
        steps=[
            # Original camera capture (DSLR/professional)
            DegradationStep(
                "gaussian_noise",  # Minimal camera sensor noise
                probability_range=(0.2, 0.5),
                parameters={"noise_range": (0.5, 3), "gray_noise_prob": 0.1},
            ),
            DegradationStep(
                "blur",  # Lens blur
                probability_range=(0.3, 0.7),
                parameters={"blur_kernel": None},  # Will be provided dynamically
            ),
            DegradationStep(
                "color_temp_shift",  # Camera white balance adjustments
                probability_range=(0.5, 0.8),
                parameter_ranges={"color_temp_shift_range": (-0.08, 0.08)},
            ),
            # Professional post-processing
            DegradationStep(
                "oversharpening",  # Professional sharpening
                probability_range=(0.6, 0.9),
                parameter_ranges={"oversharpen_strength": (1.05, 1.3)},
            ),
            # High-quality compression for sharing
            DegradationStep(
                "jpeg_compression",
                probability_range=(0.8, 1.0),
                parameter_ranges={"jpeg_range": (85, 95)},
            ),
            # User opens and re-processes
            DegradationStep(
                "exposure_error",  # User exposure adjustments
                probability_range=(0.3, 0.6),
                parameter_ranges={"exposure_factor_range": (0.8, 1.3)},
            ),
            # User sharpening (often overdone)
            DegradationStep(
                "oversharpening",
                probability_range=(0.7, 0.9),
                parameter_ranges={"oversharpen_strength": (1.2, 1.8)},
            ),
            # Platform processing and compression
            DegradationStep(
                "lens_distortion",  # Platform algorithms
                probability_range=(0.4, 0.7),
                parameter_ranges={"lens_distort_strength_range": (-0.05, 0.05)},
            ),
            DegradationStep(
                "webp_compression",  # Web platform compression
                probability_range=(0.8, 1.0),
                parameter_ranges={"webp_range": (70, 85)},
            ),
        ],
    )

    # Phone Camera Workflow with Social Media
    phone_to_social = DegradationSequence(
        name="phone_to_social",
        probability=0.25,
        repeat=1,
        repeat_probability=0.5,  # High chance of re-uploads
        steps=[
            # Phone camera artifacts (capture phase)
            DegradationStep(
                "sensor_noise",  # Phone sensor noise
                probability_range=(0.7, 1.0),
                parameter_ranges={"sensor_noise_std_range": (0.02, 0.08)},
            ),
            DegradationStep(
                "rolling_shutter",  # CMOS rolling shutter
                probability_range=(0.4, 0.8),
                parameter_ranges={"rolling_shutter_strength_range": (0.01, 0.06)},
            ),
            DegradationStep(
                "motion_blur",  # Hand shake
                probability_range=(0.2, 0.5),
                parameter_ranges={
                    "motion_blur_kernel_size": (3, 7),
                    "motion_blur_angle_range": (0, 360),
                },
            ),
            # Phone processing
            DegradationStep(
                "oversharpening",  # Heavy phone sharpening
                probability_range=(0.8, 1.0),
                parameter_ranges={"oversharpen_strength": (1.2, 1.6)},
            ),
            DegradationStep(
                "lens_distortion",  # Wide-angle phone lens
                probability_range=(0.6, 0.9),
                parameter_ranges={"lens_distort_strength_range": (0.08, 0.25)},
            ),
            DegradationStep(
                "chromatic_aberration",  # Cheap lens aberration
                probability_range=(0.5, 0.8),
            ),
            # Social media workflow
            DegradationStep(
                "resize",  # Platform resizing
                probability_range=(0.7, 1.0),
                parameters={
                    "scale_factor": RNG.get_rng().uniform(0.5, 1.2),
                    "mode": RNG.get_rng().choice(["bilinear", "bicubic", "lanczos"]),
                },
            ),
            DegradationStep(
                "oversharpening",  # Platform sharpening to improve appearance
                probability_range=(0.6, 0.9),
                parameter_ranges={"oversharpen_strength": (1.1, 1.5)},
            ),
            # Heavy social media compression
            DegradationStep(
                "webp_compression",
                probability_range=(0.9, 1.0),
                parameter_ranges={"webp_range": (60, 80)},
            ),
            # User downloads and re-uploads (repeat cycle)
            DegradationStep(
                "jpeg_compression",  # Different format compression
                probability_range=(0.3, 0.6),
                parameter_ranges={"jpeg_range": (70, 85)},
            ),
            DegradationStep(
                "oversharpening",  # User sharpening again
                probability_range=(0.5, 0.8),
                parameter_ranges={"oversharpen_strength": (1.1, 1.7)},
            ),
        ],
    )

    # Instagram/TikTok Style Processing
    social_processing = DegradationSequence(
        name="social_processing",
        probability=0.3,
        repeat=1,
        repeat_probability=0.6,  # Very common to re-upload multiple times
        steps=[
            # User preparation
            DegradationStep(
                "exposure_error",  # Brightness/contrast filters
                probability_range=(0.6, 0.9),
                parameter_ranges={"exposure_factor_range": (0.9, 1.4)},
            ),
            DegradationStep(
                "color_temp_shift",  # Color filters
                probability_range=(0.7, 0.9),
                parameter_ranges={"color_temp_shift_range": (-0.2, 0.2)},
            ),
            DegradationStep(
                "oversharpening",  # Beauty filters oversharpening
                probability_range=(0.8, 0.95),
                parameter_ranges={"oversharpen_strength": (1.3, 2.2)},
            ),
            # Platform processing
            DegradationStep(
                "aliasing",  # Heavy downsampling for performance
                probability_range=(0.4, 0.7),
                parameter_ranges={"aliasing_scale_range": (0.7, 0.9)},
            ),
            DegradationStep(
                "resize",  # Aggressive resizing for feeds
                probability_range=(0.8, 1.0),
                parameters={
                    "scale_factor": RNG.get_rng().uniform(0.6, 0.95),
                    "mode": "bilinear",  # Fast but quality sacrifice
                },
            ),
            # Platform compression + sharpening cycle
            DegradationStep(
                "webp_compression",
                probability_range=(0.9, 1.0),
                parameter_ranges={"webp_range": (50, 75)},
            ),
            DegradationStep(
                "oversharpening",  # Platform sharpening to counteract compression
                probability_range=(0.7, 0.9),
                parameter_ranges={"oversharpen_strength": (1.2, 1.8)},
            ),
            # Multiple re-upload cycles
            DegradationStep(
                "jpeg_compression",
                probability_range=(0.4, 0.7),
                parameter_ranges={"jpeg_range": (60, 80)},
            ),
            DegradationStep(
                "motion_blur",  # Compression artifacts creating motion-like patterns
                probability_range=(0.2, 0.4),
                parameter_ranges={
                    "motion_blur_kernel_size": (2, 5),
                    "motion_blur_angle_range": (0, 360),
                },
            ),
        ],
    )

    # Old Internet Image Workflow (legacy)
    legacy_internet = DegradationSequence(
        name="legacy_internet",
        probability=0.25,
        repeat=1,
        repeat_probability=0.4,
        steps=[
            # Original processing
            DegradationStep(
                "oversharpening",  # Old photo editors oversharpening
                probability_range=(0.6, 0.8),
                parameter_ranges={"oversharpen_strength": (1.1, 1.4)},
            ),
            DegradationStep(
                "gaussian_noise",  # Scan noise or original artifacts
                probability_range=(0.4, 0.7),
                parameters={"noise_range": (2, 8), "gray_noise_prob": 0.2},
            ),
            # Early JPEG compression (lower quality standards)
            DegradationStep(
                "jpeg_compression",
                probability_range=(0.8, 1.0),
                parameter_ranges={"jpeg_range": (60, 80)},
            ),
            # Multiple save/re-upload cycles
            DegradationStep(
                "jpeg_compression",
                probability_range=(0.5, 0.8),
                parameter_ranges={"jpeg_range": (50, 70)},
            ),
            DegradationStep(
                "resize",  # Platform automatic resizing
                probability_range=(0.6, 0.9),
                parameters={
                    "scale_factor": RNG.get_rng().uniform(0.8, 1.1),
                    "mode": "bilinear",
                },
            ),
            DegradationStep(
                "oversharpening",  # Platform sharpening algorithms
                probability_range=(0.5, 0.8),
                parameter_ranges={"oversharpen_strength": (1.05, 1.3)},
            ),
            # Additional compression damage
            DegradationStep(
                "jpeg_compression",
                probability_range=(0.3, 0.6),
                parameter_ranges={"jpeg_range": (40, 60)},
            ),
        ],
    )

    return [
        professional_to_internet,
        phone_to_social,
        social_processing,
        legacy_internet,
    ]


def create_enhanced_predefined_sequences() -> list[DegradationSequence]:
    """Combine original ParagonSR sequences with comprehensive workflows."""
    from traiNNer.models.paragon_sequences import create_predefined_sequences

    # Get original ParagonSR sequences
    original_sequences = create_predefined_sequences()

    # Get comprehensive combined sequences
    comprehensive_sequences = create_comprehensive_sequences()

    # Return combined list with adjusted probabilities
    return original_sequences + comprehensive_sequences
