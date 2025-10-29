"""
Comprehensive Degradation Sequence Control for ParagonSR.

This module implements realistic degradation sequences that simulate
real-world photography and internet workflows. The sequence control
system supports BOTH Real-ESRGAN degradations and ParagonSR extended
degradations for truly realistic degradation chains.

SUPPORTED DEGRADATION TYPES:

Real-ESRGAN Core Degradations:
- gaussian_noise: Apply Gaussian noise with configurable sigma range
- poisson_noise: Apply Poisson noise with configurable scale range
- blur: Apply Gaussian blur using kernel matrices
- resize: Resize images using various interpolation modes
- jpeg_compression: Apply JPEG compression artifacts

ParagonSR Extended Degradations:
- webp_compression: Modern WebP compression artifacts
- avif_compression: Next-gen AVIF compression artifacts
- heif_compression: HEIF/HEIC compression artifacts
- motion_blur: Realistic motion blur simulation
- lens_distortion: Camera lens distortion effects
- exposure_error: Exposure/brightness correction errors
- color_temp_shift: Color temperature adjustments
- sensor_noise: Camera sensor noise patterns
- rolling_shutter: CMOS rolling shutter effects
- oversharpening: Oversharpening artifact simulation
- chromatic_aberration: Lens chromatic aberration
- demosaicing: Demosaicing artifact simulation
- aliasing: Aliasing artifact simulation

The sequence system allows users to create custom workflows by combining
these degradations in realistic orders that simulate actual photography
workflows from capture → processing → compression → sharing.

Author: Philip Hofmann
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from traiNNer.models.paragon_otf_degradations import ParagonOTF
from traiNNer.utils import RNG


class DegradationStep:
    """Represents a single step in a degradation sequence."""

    def __init__(
        self,
        degradation_type: str,
        probability: float = 1.0,
        parameters: dict[str, Any] | None = None,
        probability_range: tuple[float, float] | None = None,
        parameter_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize a degradation step.

        Args:
            degradation_type: Type of degradation to apply
            probability: Fixed probability (0-1) of applying this step
            parameters: Fixed parameters for the degradation
            probability_range: Range for random probability selection
            parameter_ranges: Ranges for random parameter selection
        """
        self.degradation_type = degradation_type
        self.probability = probability
        self.parameters = parameters or {}
        self.probability_range = probability_range
        self.parameter_ranges = parameter_ranges or {}

    def should_apply(self) -> bool:
        """Determine if this step should be applied based on probability."""
        if self.probability_range:
            prob = RNG.get_rng().uniform(
                self.probability_range[0], self.probability_range[1]
            )
        else:
            prob = self.probability

        return RNG.get_rng().uniform() < prob

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters for this step, with randomization if ranges are specified."""
        params = self.parameters.copy()

        for param_name, param_range in self.parameter_ranges.items():
            if param_name not in params:
                params[param_name] = RNG.get_rng().uniform(
                    param_range[0], param_range[1]
                )

        return params


class DegradationSequence:
    """A sequence of degradation steps that can be applied in order."""

    def __init__(
        self,
        name: str,
        probability: float,
        steps: list[DegradationStep],
        repeat: int = 1,
        repeat_probability: float = 0.0,
    ) -> None:
        """Initialize a degradation sequence.

        Args:
            name: Name of the sequence
            probability: Probability of selecting this sequence
            steps: List of degradation steps
            repeat: Number of times to repeat the sequence
            repeat_probability: Probability of additional repetitions
        """
        self.name = name
        self.probability = probability
        self.steps = steps
        self.repeat = repeat
        self.repeat_probability = repeat_probability

    def should_apply(self) -> bool:
        """Determine if this sequence should be applied."""
        return RNG.get_rng().uniform() < self.probability

    def get_repeat_count(self) -> int:
        """Get the number of times to repeat this sequence."""
        count = self.repeat

        # Add random repetitions based on repeat_probability
        while RNG.get_rng().uniform() < self.repeat_probability:
            count += 1

        return count


class SequenceController:
    """Controls the application of degradation sequences."""

    def __init__(self, sequences: list[DegradationSequence]) -> None:
        """Initialize with a list of sequences.

        Args:
            sequences: List of DegradationSequence objects
        """
        self.sequences = sequences
        self.total_probability = sum(seq.probability for seq in sequences)
        self.jpeger = None  # Initialize jpeger for JPEG compression

        if self.total_probability > 1.0:
            raise ValueError(
                f"Total sequence probability {self.total_probability} exceeds 1.0"
            )

    def select_sequence(self) -> DegradationSequence | None:
        """Select a sequence based on probabilities."""
        if not self.sequences:
            return None

        rand_val = RNG.get_rng().uniform(0, self.total_probability)
        cumulative_prob = 0.0

        for sequence in self.sequences:
            cumulative_prob += sequence.probability
            if rand_val <= cumulative_prob:
                return sequence

        return None

    def apply_sequence(
        self, img_tensor: torch.Tensor, opt, sequence: DegradationSequence | None = None
    ) -> torch.Tensor:
        """Apply a degradation sequence to an image tensor."""
        if sequence is None:
            sequence = self.select_sequence()

        if sequence is None:
            return img_tensor

        current_img = img_tensor.clone()

        # Apply sequence the specified number of times
        repeat_count = sequence.get_repeat_count()
        for _ in range(repeat_count):
            current_img = self._apply_sequence_once(current_img, opt, sequence)

        return current_img

    def _apply_sequence_once(
        self, img_tensor: torch.Tensor, opt, sequence: DegradationSequence
    ) -> torch.Tensor:
        """Apply a sequence once (without repetition)."""
        current_img = img_tensor.clone()

        for step in sequence.steps:
            if step.should_apply():
                current_img = self._apply_step(current_img, opt, step)

        return current_img

    def _apply_step(
        self, img_tensor: torch.Tensor, opt, step: DegradationStep
    ) -> torch.Tensor:
        """Apply a single degradation step.

        Supports both Real-ESRGAN degradations and ParagonSR extended degradations:

        Real-ESRGAN Degradations:
        - gaussian_noise, poisson_noise, blur, resize, jpeg_compression

        ParagonSR Extended Degradations:
        - webp_compression, avif_compression, heif_compression, motion_blur,
          lens_distortion, exposure_error, color_temp_shift, sensor_noise,
          rolling_shutter, oversharpening, chromatic_aberration,
          demosaicing, aliasing
        """
        # Create a mock opt object with step-specific parameters
        step_opt = self._create_step_opt(opt, step)

        # Apply Real-ESRGAN degradations
        if step.degradation_type == "gaussian_noise":
            from traiNNer.data.degradations import random_add_gaussian_noise_pt

            return random_add_gaussian_noise_pt(
                img_tensor,
                sigma_range=step.get_parameters().get("noise_range", (0, 15)),
                clip=True,
                rounds=False,
                gray_prob=step.get_parameters().get("gray_noise_prob", 0),
            )
        elif step.degradation_type == "poisson_noise":
            from traiNNer.data.degradations import random_add_poisson_noise_pt

            return random_add_poisson_noise_pt(
                img_tensor,
                scale_range=step.get_parameters().get(
                    "poisson_scale_range", (0.01, 1.5)
                ),
                gray_prob=step.get_parameters().get("gray_noise_prob", 0),
                clip=True,
                rounds=False,
            )
        elif step.degradation_type == "blur":
            from traiNNer.utils.img_process_util import filter2d

            # Use step_opt.blur_kernel as the kernel (passed via parameters)
            kernel = step.get_parameters().get("blur_kernel")
            if kernel is not None:
                return filter2d(img_tensor, kernel)
            else:
                return img_tensor  # Skip if no kernel provided
        elif step.degradation_type == "resize":
            from traiNNer.data.degradations import resize_pt

            scale_factor = step.get_parameters().get("scale_factor", 1.0)
            mode = step.get_parameters().get("mode", "bicubic")
            return resize_pt(img_tensor, scale_factor=scale_factor, mode=mode)
        elif step.degradation_type == "jpeg_compression":
            # Use built-in jpeger
            jpeger = getattr(self, "jpeger", None)
            if jpeger is None:
                from traiNNer.utils import DiffJPEG

                self.jpeger = DiffJPEG(differentiable=False).cuda()
            quality_range = step.get_parameters().get("jpeg_range", (75, 95))
            quality = RNG.get_rng().uniform(quality_range[0], quality_range[1])
            img_clamped = torch.clamp(img_tensor, 0, 1)
            return self.jpeger(img_clamped, quality=quality)

        # Apply ParagonSR extended degradations
        elif step.degradation_type == "webp_compression":
            return ParagonOTF.apply_webp_compression(img_tensor, step_opt)
        elif step.degradation_type == "avif_compression":
            return ParagonOTF.apply_avif_compression(img_tensor, step_opt)
        elif step.degradation_type == "heif_compression":
            return ParagonOTF.apply_heif_compression(img_tensor, step_opt)
        elif step.degradation_type == "motion_blur":
            return ParagonOTF.apply_motion_blur(img_tensor, step_opt)
        elif step.degradation_type == "lens_distortion":
            return ParagonOTF.apply_lens_distortion(img_tensor, step_opt)
        elif step.degradation_type == "exposure_error":
            return ParagonOTF.apply_exposure_errors(img_tensor, step_opt)
        elif step.degradation_type == "color_temp_shift":
            return ParagonOTF.apply_color_temperature_shift(img_tensor, step_opt)
        elif step.degradation_type == "sensor_noise":
            return ParagonOTF.apply_sensor_noise(img_tensor, step_opt)
        elif step.degradation_type == "rolling_shutter":
            return ParagonOTF.apply_rolling_shutter(img_tensor, step_opt)
        elif step.degradation_type == "oversharpening":
            return ParagonOTF.apply_oversharpening(img_tensor, step_opt)
        elif step.degradation_type == "chromatic_aberration":
            return ParagonOTF.apply_chromatic_aberration(img_tensor, step_opt)
        elif step.degradation_type == "demosaicing":
            return ParagonOTF.apply_demosaicing_artifacts(img_tensor, step_opt)
        elif step.degradation_type == "aliasing":
            return ParagonOTF.apply_aliasing_artifacts(img_tensor, step_opt)
        else:
            # Fallback for unrecognized degradation types
            return img_tensor

    def _create_step_opt(self, original_opt, step: DegradationStep) -> Any:
        """Create an option object for a specific step with its parameters."""

        class StepOpt:
            def __init__(self, base_opt, step_params) -> None:
                self.base_opt = base_opt
                self.step_params = step_params

                # Copy all attributes from base opt
                for attr_name in dir(base_opt):
                    if not attr_name.startswith("_") and hasattr(base_opt, attr_name):
                        setattr(self, attr_name, getattr(base_opt, attr_name))

                # Override with step parameters
                for param_name, param_value in step_params.items():
                    setattr(self, param_name, param_value)

        return StepOpt(original_opt, step.get_parameters())


def create_predefined_sequences() -> list[DegradationSequence]:
    """Create predefined realistic degradation sequences."""

    # Internet Upload/Download Cycle Sequence
    internet_sequence = DegradationSequence(
        name="internet_upload_download",
        probability=0.25,
        repeat=1,
        repeat_probability=0.3,  # 30% chance of additional cycles
        steps=[
            # User oversharpens before upload
            DegradationStep(
                "oversharpening",
                probability_range=(0.6, 0.9),
                parameter_ranges={"oversharpen_strength": (1.1, 1.8)},
            ),
            # Color temperature adjustment during editing
            DegradationStep(
                "color_temp_shift",
                probability_range=(0.3, 0.7),
                parameter_ranges={"color_temp_shift_range": (-0.15, 0.15)},
            ),
            # Platform resizes and applies its own processing
            DegradationStep(
                "lens_distortion",
                probability_range=(0.2, 0.5),
                parameter_ranges={"lens_distort_strength_range": (-0.1, 0.1)},
            ),
            # First compression (upload)
            DegradationStep(
                "webp_compression",
                probability=1.0,  # Always apply for internet sequence
                parameter_ranges={"webp_range": (60, 85)},
            ),
            # Possible additional compression format
            DegradationStep(
                "avif_compression",
                probability_range=(0.1, 0.3),
                parameter_ranges={"avif_range": (65, 90)},
            ),
            # Download and re-upload (repeat cycle)
            DegradationStep(
                "jpeg_compression",  # Simulate different platform
                probability_range=(0.2, 0.4),
                parameter_ranges={"jpeg_range": (70, 90)},
            ),
            # Platform sharpening to counteract compression softness
            DegradationStep(
                "oversharpening",
                probability_range=(0.4, 0.8),
                parameter_ranges={"oversharpen_strength": (1.05, 1.4)},
            ),
        ],
    )

    # Phone Camera Sequence
    phone_sequence = DegradationSequence(
        name="phone_camera_capture",
        probability=0.3,
        steps=[
            # Sensor noise (always present in phone cameras)
            DegradationStep(
                "sensor_noise",
                probability_range=(0.8, 1.0),
                parameter_ranges={"sensor_noise_std_range": (0.02, 0.08)},
            ),
            # Rolling shutter effect (common in phone cameras)
            DegradationStep(
                "rolling_shutter",
                probability_range=(0.3, 0.7),
                parameter_ranges={"rolling_shutter_strength_range": (0.02, 0.08)},
            ),
            # Lens distortion (wide-angle phone lenses)
            DegradationStep(
                "lens_distortion",
                probability_range=(0.6, 0.9),
                parameter_ranges={"lens_distort_strength_range": (0.1, 0.3)},
            ),
            # Motion blur (handshake)
            DegradationStep(
                "motion_blur",
                probability_range=(0.2, 0.5),
                parameter_ranges={
                    "motion_blur_kernel_size": (3, 7),
                    "motion_blur_angle_range": (0, 360),
                },
            ),
            # Chromatic aberration
            DegradationStep(
                "chromatic_aberration",
                probability_range=(0.4, 0.8),
            ),
            # Processing artifacts
            DegradationStep(
                "oversharpening",
                probability_range=(0.7, 0.9),
                parameter_ranges={"oversharpen_strength": (1.1, 1.5)},
            ),
            # Final compression for storage/sharing
            DegradationStep(
                "heif_compression",  # HEIF is common in phones
                probability_range=(0.8, 1.0),
                parameter_ranges={"heif_range": (75, 95)},
            ),
        ],
    )

    # DSLR/Professional Camera Sequence
    dslr_sequence = DegradationSequence(
        name="dslr_professional",
        probability=0.2,
        steps=[
            # Minimal sensor noise (better sensors)
            DegradationStep(
                "sensor_noise",
                probability_range=(0.3, 0.6),
                parameter_ranges={"sensor_noise_std_range": (0.005, 0.03)},
            ),
            # Minimal rolling shutter
            DegradationStep(
                "rolling_shutter",
                probability_range=(0.1, 0.3),
                parameter_ranges={"rolling_shutter_strength_range": (0.005, 0.02)},
            ),
            # Professional lens distortion (still present but less)
            DegradationStep(
                "lens_distortion",
                probability_range=(0.4, 0.7),
                parameter_ranges={"lens_distort_strength_range": (0.02, 0.1)},
            ),
            # Professional post-processing
            DegradationStep(
                "oversharpening",
                probability_range=(0.5, 0.8),
                parameter_ranges={"oversharpen_strength": (1.05, 1.3)},
            ),
            DegradationStep(
                "color_temp_shift",
                probability_range=(0.4, 0.7),
                parameter_ranges={"color_temp_shift_range": (-0.1, 0.1)},
            ),
            # High-quality compression
            DegradationStep(
                "jpeg_compression",
                probability_range=(0.8, 1.0),
                parameter_ranges={"jpeg_range": (85, 98)},
            ),
        ],
    )

    # Social Media Upload Sequence
    social_sequence = DegradationSequence(
        name="social_media_upload",
        probability=0.25,
        repeat=1,
        repeat_probability=0.4,  # Multiple re-uploads common
        steps=[
            # User preparation
            DegradationStep(
                "oversharpening",
                probability_range=(0.7, 0.95),
                parameter_ranges={
                    "oversharpen_strength": (1.2, 2.0)  # Often overdone
                },
            ),
            # Platform processing
            DegradationStep(
                "lens_distortion",
                probability_range=(0.3, 0.6),
                parameter_ranges={"lens_distort_strength_range": (-0.05, 0.05)},
            ),
            # Heavy compression for fast loading
            DegradationStep(
                "webp_compression",
                probability_range=(0.9, 1.0),
                parameter_ranges={"webp_range": (50, 80)},
            ),
            # Additional compression when re-uploading
            DegradationStep(
                "jpeg_compression",
                probability_range=(0.4, 0.7),
                parameter_ranges={"jpeg_range": (60, 85)},
            ),
            # Platform sharpening to improve appearance
            DegradationStep(
                "oversharpening",
                probability_range=(0.6, 0.9),
                parameter_ranges={"oversharpen_strength": (1.1, 1.6)},
            ),
        ],
    )

    return [internet_sequence, phone_sequence, dslr_sequence, social_sequence]
