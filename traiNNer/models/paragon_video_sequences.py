"""
Platform-Specific Video Degradation Sequences for ParagonSR.

This module provides realistic degradation sequences tailored to specific
video platforms and use cases:
- YouTube (H.264, adaptive quality)
- TikTok/Instagram Reels (aggressive compression, filters)
- Streaming services (Netflix/Prime - H.265/AV1, high quality)
- DVD/Anime rips (MPEG-2 → H.264, interlacing artifacts)
- Social media general (multi-platform cycles)

Author: Philip Hofmann
"""

from traiNNer.models.paragon_sequences import (
    DegradationSequence,
    DegradationStep,
)
from traiNNer.utils import RNG


def create_video_platform_sequences() -> list[DegradationSequence]:
    """Create platform-specific video degradation sequences."""

    # =========================================================================
    # YOUTUBE SEQUENCE
    # Simulates YouTube's H.264 encoding with adaptive bitrate
    # CRF range reflects quality tiers (1080p HD, 720p, 480p)
    # =========================================================================

    youtube_sequence = DegradationSequence(
        name="youtube_video",
        probability=0.15,
        repeat=1,
        steps=[
            # User pre-upload processing
            DegradationStep(
                "oversharpening",
                probability_range=(0.7, 0.9),
                parameter_ranges={"oversharpen_strength": (1.1, 1.5)},
            ),
            DegradationStep(
                "color_temp_shift",
                probability_range=(0.4, 0.7),
                parameter_ranges={"color_temp_shift_range": (-0.1, 0.1)},
            ),
            # YouTube H.264 encoding (adaptive quality)
            DegradationStep(
                "video_compression",
                probability=1.0,
                parameters={
                    "video_compress_prob": 1.0,
                    "video_codecs": ["h264"],
                    "video_crf_range": (23, 35),  # 23=1080p HD, 35=480p
                    "video_presets": ["medium", "fast"],
                },
            ),
            # Post-compression artifacts
            DegradationStep(
                "block_artifacts",
                probability_range=(0.5, 0.8),
                parameters={
                    "block_artifact_prob": 1.0,
                    "block_strength_range": (8, 16),
                },
            ),
            DegradationStep(
                "color_banding",
                probability_range=(0.4, 0.7),
                parameters={
                    "banding_prob": 1.0,
                    "banding_bit_range": (6, 8),
                },
            ),
            DegradationStep(
                "ringing",
                probability_range=(0.3, 0.6),
                parameters={
                    "ringing_prob": 1.0,
                    "ringing_strength_range": (0.02, 0.08),
                },
            ),
            # Platform sharpening filter
            DegradationStep(
                "oversharpening",
                probability_range=(0.6, 0.9),
                parameter_ranges={"oversharpen_strength": (1.05, 1.3)},
            ),
        ],
    )

    # =========================================================================
    # TIKTOK / INSTAGRAM REELS SEQUENCE
    # Very aggressive compression for mobile bandwidth
    # Heavy user filters and beauty mode processing
    # Common screen recording artifacts from re-uploads
    # =========================================================================

    tiktok_sequence = DegradationSequence(
        name="tiktok_shortform",
        probability=0.15,
        repeat=1,
        repeat_probability=0.5,  # Often re-uploaded/screen-recorded
        steps=[
            # Heavy user filters (beauty mode, color grading)
            DegradationStep(
                "exposure_error",
                probability_range=(0.7, 0.95),
                parameter_ranges={"exposure_factor_range": (0.85, 1.4)},
            ),
            DegradationStep(
                "color_temp_shift",
                probability_range=(0.8, 0.95),
                parameter_ranges={"color_temp_shift_range": (-0.25, 0.25)},
            ),
            DegradationStep(
                "oversharpening",
                probability_range=(0.85, 0.98),
                parameter_ranges={
                    "oversharpen_strength": (1.3, 2.5)
                },  # Very aggressive!
            ),
            # Platform aggressive compression (bandwidth optimization)
            DegradationStep(
                "video_compression",
                probability=1.0,
                parameters={
                    "video_compress_prob": 1.0,
                    "video_codecs": ["h264"],
                    "video_crf_range": (28, 40),  # Lower quality for bandwidth
                    "video_presets": ["veryfast", "ultrafast"],
                },
            ),
            # Heavy artifacts from aggressive compression
            DegradationStep(
                "block_artifacts",
                probability_range=(0.7, 0.95),
                parameters={
                    "block_artifact_prob": 1.0,
                    "block_strength_range": (12, 24),
                },
            ),
            DegradationStep(
                "color_banding",
                probability_range=(0.6, 0.9),
                parameters={
                    "banding_prob": 1.0,
                    "banding_bit_range": (5, 7),
                },
            ),
            # Screen recording artifacts (common for re-uploads)
            DegradationStep(
                "aliasing",
                probability_range=(0.4, 0.7),
                parameter_ranges={"aliasing_scale_range": (0.6, 0.85)},
            ),
            DegradationStep(
                "motion_blur",
                probability_range=(0.3, 0.6),
                parameters={
                    "motion_blur_prob": 1.0,
                    "motion_blur_kernel_size": (3, 7),
                    "motion_blur_angle_range": (0, 360),
                },
            ),
        ],
    )

    # =========================================================================
    # STREAMING SERVICES (Netflix, Prime, Disney+)
    # High-quality H.265/AV1 encoding
    # Minimal artifacts but still present
    # Sometimes screen-captured for piracy
    # =========================================================================

    streaming_sequence = DegradationSequence(
        name="streaming_service",
        probability=0.15,
        steps=[
            # High-quality source (minimal noise)
            DegradationStep(
                "gaussian_noise",
                probability_range=(0.2, 0.4),
                parameters={"noise_range": (0.5, 2), "gray_noise_prob": 0},
            ),
            # Modern codec with adaptive bitrate (H.265/AV1)
            DegradationStep(
                "video_compression",
                probability=1.0,
                parameters={
                    "video_compress_prob": 1.0,
                    "video_codecs": ["h265", "h264"],  # Modern codecs
                    "video_crf_range": (18, 28),  # Higher quality
                    "video_presets": ["slow", "medium"],  # Better encoding
                },
            ),
            # Minimal artifacts (high quality)
            DegradationStep(
                "block_artifacts",
                probability_range=(0.3, 0.6),
                parameters={
                    "block_artifact_prob": 1.0,
                    "block_strength_range": (6, 12),
                },
            ),
            DegradationStep(
                "color_banding",
                probability_range=(0.3, 0.5),
                parameters={
                    "banding_prob": 1.0,
                    "banding_bit_range": (7, 8),
                },
            ),
            # Sometimes screen-captured (piracy)
            DegradationStep(
                "motion_blur",
                probability_range=(0.1, 0.3),
                parameters={
                    "motion_blur_prob": 1.0,
                    "motion_blur_kernel_size": (2, 5),
                    "motion_blur_angle_range": (0, 360),
                },
            ),
            DegradationStep(
                "exposure_error",
                probability_range=(0.2, 0.4),
                parameter_ranges={"exposure_factor_range": (0.9, 1.1)},
            ),
        ],
    )

    # =========================================================================
    # DVD RIP / ANIME STREAMING
    # Legacy MPEG-2 → H.264 re-encoding
    # Interlacing artifacts from old capture methods
    # Heavy ringing from multiple compression passes
    # =========================================================================

    dvdrip_sequence = DegradationSequence(
        name="dvdrip_anime",
        probability=0.1,
        steps=[
            # Original capture artifacts
            DegradationStep(
                "sensor_noise",
                probability_range=(0.6, 0.9),
                parameter_ranges={"sensor_noise_std_range": (0.01, 0.05)},
            ),
            # MPEG-2 compression (DVD source)
            # Note: FFmpeg may not support mpeg2, fallback to h264
            DegradationStep(
                "video_compression",
                probability=1.0,
                parameters={
                    "video_compress_prob": 1.0,
                    "video_codecs": ["mpeg2", "h264"],  # Try MPEG-2, fallback to H.264
                    "video_crf_range": (18, 28),
                    "video_presets": ["medium"],
                },
            ),
            # Re-encode to H.264 (rip process)
            DegradationStep(
                "video_compression",
                probability=1.0,
                parameters={
                    "video_compress_prob": 1.0,
                    "video_codecs": ["h264"],
                    "video_crf_range": (20, 30),
                    "video_presets": ["slow", "medium"],
                },
            ),
            # Rip artifacts (double compression)
            DegradationStep(
                "block_artifacts",
                probability_range=(0.7, 0.9),
                parameters={
                    "block_artifact_prob": 1.0,
                    "block_strength_range": (10, 20),
                },
            ),
            DegradationStep(
                "ringing",
                probability_range=(0.6, 0.9),
                parameters={
                    "ringing_prob": 1.0,
                    "ringing_strength_range": (0.05, 0.15),
                },
            ),
            # Fan subber/encoder sharpening (common in anime rips)
            DegradationStep(
                "oversharpening",
                probability_range=(0.7, 0.9),
                parameter_ranges={"oversharpen_strength": (1.2, 1.8)},
            ),
        ],
    )

    # =========================================================================
    # SOCIAL MEDIA GENERAL (Multi-platform)
    # Simulates content uploaded to multiple platforms
    # Each with their own compression + sharpening
    # Common workflow: phone → Instagram → screenshot → Twitter
    # =========================================================================

    social_multi_sequence = DegradationSequence(
        name="social_multi_platform",
        probability=0.15,
        repeat=1,
        repeat_probability=0.7,  # Very common to go through multiple platforms
        steps=[
            # User processing
            DegradationStep(
                "oversharpening",
                probability_range=(0.8, 0.95),
                parameter_ranges={"oversharpen_strength": (1.2, 2.0)},
            ),
            DegradationStep(
                "color_temp_shift",
                probability_range=(0.5, 0.8),
                parameter_ranges={"color_temp_shift_range": (-0.15, 0.15)},
            ),
            # First platform compression (Instagram/Facebook)
            DegradationStep(
                "video_compression",
                probability=1.0,
                parameters={
                    "video_compress_prob": 1.0,
                    "video_codecs": ["h264"],
                    "video_crf_range": (25, 35),
                    "video_presets": ["fast", "medium"],
                },
            ),
            # Screenshot/download
            DegradationStep(
                "aliasing",
                probability_range=(0.5, 0.8),
                parameter_ranges={"aliasing_scale_range": (0.7, 0.9)},
            ),
            # Second platform re-upload (Twitter/Reddit)
            DegradationStep(
                "video_compression",
                probability_range=(0.6, 0.9),
                parameters={
                    "video_compress_prob": 1.0,
                    "video_codecs": ["h264"],
                    "video_crf_range": (28, 38),
                    "video_presets": ["veryfast", "ultrafast"],
                },
            ),
            # Accumulated artifacts
            DegradationStep(
                "block_artifacts",
                probability_range=(0.7, 0.95),
                parameters={
                    "block_artifact_prob": 1.0,
                    "block_strength_range": (14, 26),
                },
            ),
            DegradationStep(
                "color_banding",
                probability_range=(0.6, 0.9),
                parameters={
                    "banding_prob": 1.0,
                    "banding_bit_range": (5, 7),
                },
            ),
            # Platform sharpening algorithms
            DegradationStep(
                "oversharpening",
                probability_range=(0.7, 0.9),
                parameter_ranges={"oversharpen_strength": (1.1, 1.6)},
            ),
        ],
    )

    return [
        youtube_sequence,
        tiktok_sequence,
        streaming_sequence,
        dvdrip_sequence,
        social_multi_sequence,
    ]
