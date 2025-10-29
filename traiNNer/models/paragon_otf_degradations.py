"""
Extended On-The-Fly (OTF) degradation functions for ParagonSR.

These functions were added by Philip Hofmann to enhance the Real-ESRGAN
pipeline with modern compression artifacts and additional realistic
degradations. They are intended to be used by the RealESRGANModel
through the ParagonOTF class.

Author: Philip Hofmann
"""

import io
import math
import random
from typing import Any

import numpy as np
import torch
from PIL import Image

# Optional AVIF and HEIF support
try:
    from pillow_avif import AvifImagePlugin
except ImportError:
    AvifImagePlugin = None

try:
    from pillow_heif import HeifImagePlugin
except ImportError:
    HeifImagePlugin = None

from traiNNer.utils import RNG


class ParagonOTF:
    """Container for static OTF degradation methods."""

    @staticmethod
    def apply_webp_compression(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply WebP compression with probability and quality range from ``opt``."""
        if not hasattr(opt, "webp_prob") or not hasattr(opt, "webp_range"):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.webp_prob:
            return img_tensor

        quality = RNG.get_rng().uniform(opt.webp_range[0], opt.webp_range[1])
        batch_size = img_tensor.size(0)
        compressed = []

        for i in range(batch_size):
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8").transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img[:, :, 0]
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="WEBP", quality=int(quality))
            buffer.seek(0)
            comp = Image.open(buffer).convert("RGB")
            comp_tensor = torch.from_numpy(np.array(comp)).float() / 255.0
            comp_tensor = comp_tensor.permute(2, 0, 1)
            compressed.append(comp_tensor)

        return torch.stack(compressed, dim=0).to(img_tensor.device)

    @staticmethod
    def apply_avif_compression(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply AVIF compression with probability and quality range from ``opt``."""
        if not hasattr(opt, "avif_prob") or not hasattr(opt, "avif_range"):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.avif_prob:
            return img_tensor
        if AvifImagePlugin is None:
            # AVIF support not available – fall back to original tensor
            return img_tensor

        quality = RNG.get_rng().uniform(opt.avif_range[0], opt.avif_range[1])
        batch_size = img_tensor.size(0)
        compressed = []

        for i in range(batch_size):
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8").transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img[:, :, 0]
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="AVIF", quality=int(quality))
            buffer.seek(0)
            comp = Image.open(buffer).convert("RGB")
            comp_tensor = torch.from_numpy(np.array(comp)).float() / 255.0
            comp_tensor = comp_tensor.permute(2, 0, 1)
            compressed.append(comp_tensor)

        return torch.stack(compressed, dim=0).to(img_tensor.device)

    @staticmethod
    def apply_heif_compression(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply HEIF compression with probability and quality range from ``opt``."""
        if not hasattr(opt, "heif_prob") or not hasattr(opt, "heif_range"):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.heif_prob:
            return img_tensor
        if HeifImagePlugin is None:
            # HEIF support not available – fall back to original tensor
            return img_tensor

        quality = RNG.get_rng().uniform(opt.heif_range[0], opt.heif_range[1])
        batch_size = img_tensor.size(0)
        compressed = []

        for i in range(batch_size):
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8").transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img[:, :, 0]
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="HEIF", quality=int(quality))
            buffer.seek(0)
            comp = Image.open(buffer).convert("RGB")
            comp_tensor = torch.from_numpy(np.array(comp)).float() / 255.0
            comp_tensor = comp_tensor.permute(2, 0, 1)
            compressed.append(comp_tensor)

        return torch.stack(compressed, dim=0).to(img_tensor.device)

    @staticmethod
    def apply_motion_blur(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply motion blur with configurable angle and kernel size."""
        if not hasattr(opt, "motion_blur_prob"):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.motion_blur_prob:
            return img_tensor

        kernel_size = random.randint(
            opt.motion_blur_kernel_size[0], opt.motion_blur_kernel_size[1]
        )
        angle = RNG.get_rng().uniform(
            opt.motion_blur_angle_range[0], opt.motion_blur_angle_range[1]
        )

        # Create motion blur kernel
        kernel = ParagonOTF._create_motion_blur_kernel(kernel_size, angle)
        kernel = kernel.to(img_tensor.device)
        kernel = kernel.repeat(img_tensor.size(1), 1, 1, 1)

        # Apply convolution
        return torch.nn.functional.conv2d(
            img_tensor, kernel, padding=kernel_size // 2, groups=img_tensor.size(1)
        )

    @staticmethod
    def _create_motion_blur_kernel(kernel_size: int, angle: float) -> torch.Tensor:
        """Create a motion blur kernel."""
        # Create line coordinates
        center = kernel_size // 2
        cos_angle = math.cos(math.radians(angle))
        sin_angle = math.sin(math.radians(angle))

        kernel = torch.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                # Check if point is on the line
                if abs(x * cos_angle + y * sin_angle) < 0.5:
                    kernel[i, j] = 1.0

        # Normalize
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def apply_lens_distortion(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply barrel/pincushion lens distortion."""
        if not hasattr(opt, "lens_distort_prob") or not hasattr(
            opt, "lens_distort_strength_range"
        ):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.lens_distort_prob:
            return img_tensor

        strength = RNG.get_rng().uniform(
            opt.lens_distort_strength_range[0], opt.lens_distort_strength_range[1]
        )

        batch_size, channels, height, width = img_tensor.shape

        # Create coordinate grid
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, height, device=img_tensor.device),
            torch.linspace(-1, 1, width, device=img_tensor.device),
            indexing="ij",
        )

        # Apply barrel distortion: r_distorted = r * (1 + k1 * r^2)
        r = torch.sqrt(grid_x**2 + grid_y**2)
        r_distorted = r * (1 + strength * r**2)

        # Avoid division by zero
        r_distorted[r == 0] = 0

        # Normalize back to [-1, 1]
        r[r == 0] = 1e-6
        grid_x_distorted = grid_x * (r_distorted / r)
        grid_y_distorted = grid_y * (r_distorted / r)

        # Create sampling grid
        grid = torch.stack([grid_x_distorted, grid_y_distorted], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Sample with distorted coordinates
        return torch.nn.functional.grid_sample(
            img_tensor,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

    @staticmethod
    def apply_exposure_errors(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply exposure variations and clipping."""
        if not hasattr(opt, "exposure_prob") or not hasattr(
            opt, "exposure_factor_range"
        ):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.exposure_prob:
            return img_tensor

        factor = RNG.get_rng().uniform(
            opt.exposure_factor_range[0], opt.exposure_factor_range[1]
        )

        # Apply exposure change
        img_tensor = img_tensor * factor

        # Clip to [0, 1]
        return torch.clamp(img_tensor, 0, 1)

    @staticmethod
    def apply_color_temperature_shift(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply color temperature shifts."""
        if not hasattr(opt, "color_temp_prob") or not hasattr(
            opt, "color_temp_shift_range"
        ):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.color_temp_prob:
            return img_tensor

        if img_tensor.size(1) != 3:
            return img_tensor

        shift = RNG.get_rng().uniform(
            opt.color_temp_shift_range[0], opt.color_temp_shift_range[1]
        )

        # Split channels
        r, g, b = img_tensor[:, 0:1], img_tensor[:, 1:2], img_tensor[:, 2:3]

        # Apply temperature shift
        # Positive values = warmer (more red/yellow)
        # Negative values = cooler (more blue)
        if shift > 0:
            r = r * (1 + shift * 0.3)
            g = g * (1 + shift * 0.1)
        else:
            b = b * (1 - shift * 0.3)
            g = g * (1 - shift * 0.1)

        return torch.clamp(torch.cat([r, g, b], dim=1), 0, 1)

    @staticmethod
    def apply_sensor_noise(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply sensor-specific noise patterns."""
        if not hasattr(opt, "sensor_noise_prob") or not hasattr(
            opt, "sensor_noise_std_range"
        ):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.sensor_noise_prob:
            return img_tensor

        noise_std = RNG.get_rng().uniform(
            opt.sensor_noise_std_range[0], opt.sensor_noise_std_range[1]
        )

        # Generate Gaussian noise
        noise = torch.randn_like(img_tensor) * noise_std

        # Add noise and clamp
        return torch.clamp(img_tensor + noise, 0, 1)

    @staticmethod
    def apply_rolling_shutter(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply rolling shutter distortion."""
        if not hasattr(opt, "rolling_shutter_prob") or not hasattr(
            opt, "rolling_shutter_strength_range"
        ):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.rolling_shutter_prob:
            return img_tensor

        strength = RNG.get_rng().uniform(
            opt.rolling_shutter_strength_range[0], opt.rolling_shutter_strength_range[1]
        )

        batch_size, channels, height, width = img_tensor.shape

        # Create slant based on motion
        slant = strength * height / width

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=img_tensor.device),
            torch.linspace(-1, 1, width, device=img_tensor.device),
            indexing="ij",
        )

        # Apply slant
        grid_x_distorted = grid_x + slant * grid_y

        grid = torch.stack([grid_x_distorted, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Sample with distorted coordinates
        return torch.nn.functional.grid_sample(
            img_tensor,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

    @staticmethod
    def apply_oversharpening(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply oversharpening artifact based on ``opt.oversharpen_prob``."""
        if not hasattr(opt, "oversharpen_prob") or not hasattr(
            opt, "oversharpen_strength"
        ):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.oversharpen_prob:
            return img_tensor

        strength = RNG.get_rng().uniform(
            opt.oversharpen_strength[0], opt.oversharpen_strength[1]
        )
        # Simple box blur as placeholder for Gaussian blur
        kernel_size = 5
        padding = kernel_size // 2
        weight = torch.ones(
            1, 1, kernel_size, kernel_size, device=img_tensor.device
        ) / (kernel_size**2)
        weight = weight.repeat(img_tensor.size(1), 1, 1, 1)
        blurred = torch.nn.functional.conv2d(
            img_tensor, weight, padding=padding, groups=img_tensor.size(1)
        )
        details = img_tensor - blurred
        oversharpened = img_tensor + details * strength
        return torch.clamp(oversharpened, 0, 1)

    @staticmethod
    def apply_chromatic_aberration(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply chromatic aberration artifact based on ``opt.chromatic_aberration_prob``."""
        if not hasattr(opt, "chromatic_aberration_prob"):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.chromatic_aberration_prob:
            return img_tensor

        batch_size, channels, _, _ = img_tensor.shape
        if channels != 3:
            return img_tensor

        r = img_tensor[:, 0:1, :, :]
        g = img_tensor[:, 1:2, :, :]
        b = img_tensor[:, 2:3, :, :]

        scale_r = 1.001
        scale_b = 0.999

        theta_r = torch.tensor(
            [[[scale_r, 0, 0], [0, scale_r, 0]]],
            dtype=torch.float32,
            device=img_tensor.device,
        ).repeat(batch_size, 1, 1)
        grid_r = torch.nn.functional.affine_grid(theta_r, r.size(), align_corners=False)
        r_scaled = torch.nn.functional.grid_sample(
            r, grid_r, align_corners=False, mode="bilinear"
        )

        theta_b = torch.tensor(
            [[[scale_b, 0, 0], [0, scale_b, 0]]],
            dtype=torch.float32,
            device=img_tensor.device,
        ).repeat(batch_size, 1, 1)
        grid_b = torch.nn.functional.affine_grid(theta_b, b.size(), align_corners=False)
        b_scaled = torch.nn.functional.grid_sample(
            b, grid_b, align_corners=False, mode="bilinear"
        )

        return torch.clamp(torch.cat([r_scaled, g, b_scaled], dim=1), 0, 1)

    @staticmethod
    def apply_demosaicing_artifacts(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply demosaicing artifacts based on ``opt.demosaic_prob``."""
        if not hasattr(opt, "demosaic_prob"):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.demosaic_prob:
            return img_tensor

        import cv2  # Imported lazily

        batch_size = img_tensor.size(0)
        processed = []

        for i in range(batch_size):
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8").transpose(1, 2, 0)
            h, w, _ = img.shape
            bayer = np.zeros((h, w), dtype=np.uint8)
            bayer[0::2, 0::2] = img[0::2, 0::2, 2]  # R
            bayer[0::2, 1::2] = img[0::2, 1::2, 1]  # G
            bayer[1::2, 0::2] = img[1::2, 0::2, 1]  # G
            bayer[1::2, 1::2] = img[1::2, 1::2, 0]  # B
            demosaiced = cv2.demosaicing(bayer, cv2.COLOR_BAYER_BG2BGR)
            demosaiced_tensor = torch.from_numpy(demosaiced).float() / 255.0
            demosaiced_tensor = demosaiced_tensor.permute(2, 0, 1)
            processed.append(demosaiced_tensor)

        return torch.stack(processed, dim=0).to(img_tensor.device)

    @staticmethod
    def apply_aliasing_artifacts(img_tensor: torch.Tensor, opt) -> torch.Tensor:
        """Apply aliasing artifacts based on ``opt.aliasing_prob``."""
        if not hasattr(opt, "aliasing_prob") or not hasattr(
            opt, "aliasing_scale_range"
        ):
            return img_tensor
        if RNG.get_rng().uniform() >= opt.aliasing_prob:
            return img_tensor

        scale = RNG.get_rng().uniform(
            opt.aliasing_scale_range[0], opt.aliasing_scale_range[1]
        )
        _, _, h, w = img_tensor.shape
        down = torch.nn.functional.interpolate(
            img_tensor, size=(int(h * scale), int(w * scale)), mode="nearest"
        )
        up = torch.nn.functional.interpolate(down, size=(h, w), mode="nearest")
        return up
