#!/usr/bin/env python3
"""
Create OTF-degraded validation dataset for ParagonSR training.

Reads a training configuration YAML file and replicates the EXACT degradation pipeline
used during training (RealESRGAN + ParagonOTF).

Usage:
    python create_validation_otf.py \\
        --input datasets/val/hr \\
        --output datasets/val/lr_showcase \\
        --config options/train/ParagonSR2/2xS_real_photography_showcase.yml \\
        --count 100
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from traiNNer.data.degradations import (
    circular_lowpass_kernel,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    random_mixed_kernels,
    resize_pt,
)
from traiNNer.models.paragon_otf_degradations import ParagonOTF
from traiNNer.utils import RNG, DiffJPEG
from traiNNer.utils.img_process_util import USMSharp, filter2d


class ConfigOpt:
    """Helper to access config dict as attributes."""

    def __init__(self, config_dict) -> None:
        self._config = config_dict
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigOpt(v))
            else:
                setattr(self, k, v)

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'ConfigOpt' object has no attribute '{name}'")

    def get(self, key, default=None):
        return self._config.get(key, default)


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


class DegradationPipeline:
    def __init__(self, opt, device="cuda") -> None:
        self.opt = opt
        self.device = device
        self.jpeger = DiffJPEG(differentiable=False).to(device)
        self.usm_sharpener = USMSharp().to(device)

        # Dataset options are usually in datasets.train
        self.ds_opt = opt.datasets.train
        # Flatten dataset options into main opt for easier access in degradation functions
        # (RealESRGANModel logic often assumes these are on self.opt)
        for k, v in self.ds_opt._config.items():
            if not hasattr(self.opt, k):
                setattr(self.opt, k, v)

        # Explicitly disable augmentations for validation
        # These are for training only and would cause HR/LR mismatch in validation
        if hasattr(self.opt, "use_hflip"):
            self.opt.use_hflip = False
        if hasattr(self.opt, "use_rot"):
            self.opt.use_rot = False

        # Pre-calculate kernels if needed, or generate on the fly
        # RealESRGAN generates kernels per sample in the dataset __getitem__
        # We will generate them on the fly here to keep it simple and consistent

    def generate_kernels(self):
        # 1st degradation kernels
        kernel_range = self.opt.datasets.train.get("kernel_range", [7, 21])
        kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))

        sinc_prob = self.opt.datasets.train.get("sinc_prob", 0.1)
        if RNG.get_rng().uniform() < sinc_prob:
            if kernel_size < 13:
                omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            else:
                omega_c = RNG.get_rng().uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.opt.datasets.train.kernel_list,
                self.opt.datasets.train.kernel_prob,
                kernel_size,
                self.opt.datasets.train.blur_sigma,
                self.opt.datasets.train.blur_sigma,
                (-math.pi, math.pi),
                self.opt.datasets.train.betag_range,
                self.opt.datasets.train.betap_range,
                noise_range=None,
            )
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel = torch.FloatTensor(kernel).to(self.device).unsqueeze(0).unsqueeze(0)

        # 2nd degradation kernels
        kernel_range2 = self.opt.datasets.train.get(
            "kernel_range2", self.opt.datasets.train.get("kernel_range", [7, 21])
        )
        kernel_size2 = random.choice(range(kernel_range2[0], kernel_range2[1] + 1, 2))

        sinc_prob2 = self.opt.datasets.train.get("sinc_prob2", 0.1)
        if RNG.get_rng().uniform() < sinc_prob2:
            if kernel_size2 < 13:
                omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            else:
                omega_c = RNG.get_rng().uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size2, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.opt.datasets.train.get(
                    "kernel_list2", self.opt.datasets.train.get("kernel_list")
                ),
                self.opt.datasets.train.get(
                    "kernel_prob2", self.opt.datasets.train.get("kernel_prob")
                ),
                kernel_size2,
                self.opt.datasets.train.get(
                    "blur_sigma2", self.opt.datasets.train.get("blur_sigma")
                ),
                self.opt.datasets.train.get(
                    "blur_sigma2", self.opt.datasets.train.get("blur_sigma")
                ),
                (-math.pi, math.pi),
                self.opt.datasets.train.get(
                    "betag_range2", self.opt.datasets.train.get("betag_range")
                ),
                self.opt.datasets.train.get(
                    "betap_range2", self.opt.datasets.train.get("betap_range")
                ),
                noise_range=None,
            )
        pad_size2 = (21 - kernel_size2) // 2
        kernel2 = np.pad(kernel2, ((pad_size2, pad_size2), (pad_size2, pad_size2)))
        kernel2 = torch.FloatTensor(kernel2).to(self.device).unsqueeze(0).unsqueeze(0)

        # Sinc kernel
        final_sinc_prob = self.opt.datasets.train.get("final_sinc_prob", 0.8)
        if RNG.get_rng().uniform() < final_sinc_prob:
            final_kernel_range = self.opt.datasets.train.get(
                "final_kernel_range", [7, 21]
            )
            kernel_size = random.choice(
                range(final_kernel_range[0], final_kernel_range[1] + 1, 2)
            )
            omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = (
                torch.FloatTensor(sinc_kernel).to(self.device).unsqueeze(0).unsqueeze(0)
            )
        else:
            sinc_kernel = torch.zeros(1, 1, 21, 21).to(self.device)
            sinc_kernel[0, 0, 10, 10] = 1

        return kernel, kernel2, sinc_kernel

    def _apply_chromatic_aberration(self, img_tensor):
        # Simplified version of the one in RealESRGANModel
        if not hasattr(self.opt, "chromatic_aberration_prob"):
            return img_tensor
        if RNG.get_rng().uniform() >= self.opt.chromatic_aberration_prob:
            return img_tensor

        batch_size, channels, _, _ = img_tensor.shape
        if channels != 3:
            return img_tensor

        r, g, b = img_tensor[:, 0:1], img_tensor[:, 1:2], img_tensor[:, 2:3]
        scale_r, scale_b = 1.001, 0.999

        theta_r = torch.tensor(
            [[[scale_r, 0, 0], [0, scale_r, 0]]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(batch_size, 1, 1)
        grid_r = F.affine_grid(theta_r, r.size(), align_corners=False)
        r_scaled = F.grid_sample(r, grid_r, align_corners=False, mode="bilinear")

        theta_b = torch.tensor(
            [[[scale_b, 0, 0], [0, scale_b, 0]]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(batch_size, 1, 1)
        grid_b = F.affine_grid(theta_b, b.size(), align_corners=False)
        b_scaled = F.grid_sample(b, grid_b, align_corners=False, mode="bilinear")

        return torch.clamp(torch.cat([r_scaled, g, b_scaled], dim=1), 0, 1)

    def _apply_demosaicing(self, img_tensor):
        if not hasattr(self.opt, "demosaic_prob"):
            return img_tensor
        if RNG.get_rng().uniform() >= self.opt.demosaic_prob:
            return img_tensor

        # Simple simulation if cv2 fails or for speed, but let's try to match logic
        # For validation script, we can do it per image on CPU if needed, but tensor is better
        # The model implementation uses cv2 on CPU.
        processed = []
        for i in range(img_tensor.size(0)):
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8").transpose(1, 2, 0)
            h, w = img.shape[:2]
            bayer = np.zeros((h, w), dtype=np.uint8)
            bayer[0::2, 0::2] = img[0::2, 0::2, 2]  # R
            bayer[0::2, 1::2] = img[0::2, 1::2, 1]  # G
            bayer[1::2, 0::2] = img[1::2, 0::2, 1]  # G
            bayer[1::2, 1::2] = img[1::2, 1::2, 0]  # B

            try:
                demosaiced = cv2.demosaicing(bayer, cv2.COLOR_BAYER_BG2BGR)
                t = torch.from_numpy(demosaiced).float() / 255.0
                processed.append(t.permute(2, 0, 1))
            except:
                processed.append(img_tensor[i].cpu())

        return torch.stack(processed).to(self.device)

    def _apply_aliasing(self, img_tensor):
        if not hasattr(self.opt, "aliasing_prob"):
            return img_tensor
        if RNG.get_rng().uniform() >= self.opt.aliasing_prob:
            return img_tensor

        scale = RNG.get_rng().uniform(
            self.opt.aliasing_scale_range[0], self.opt.aliasing_scale_range[1]
        )
        _, _, h, w = img_tensor.shape
        down = F.interpolate(
            img_tensor, size=(int(h * scale), int(w * scale)), mode="nearest"
        )
        up = F.interpolate(down, size=(h, w), mode="nearest")
        return up

    def _apply_oversharpening(self, img_tensor):
        if not hasattr(self.opt, "oversharpen_prob"):
            return img_tensor
        if RNG.get_rng().uniform() >= self.opt.oversharpen_prob:
            return img_tensor

        strength = RNG.get_rng().uniform(
            self.opt.oversharpen_strength[0], self.opt.oversharpen_strength[1]
        )
        # Box blur
        k = 5
        weight = torch.ones(1, 1, k, k, device=self.device) / (k * k)
        weight = weight.repeat(img_tensor.size(1), 1, 1, 1)
        blurred = F.conv2d(
            img_tensor, weight, padding=k // 2, groups=img_tensor.size(1)
        )
        details = img_tensor - blurred
        return torch.clamp(img_tensor + details * strength, 0, 1)

    def process(self, gt_tensor):
        # gt_tensor: (B, C, H, W)
        kernel1, kernel2, sinc_kernel = self.generate_kernels()
        ori_h, ori_w = gt_tensor.shape[2:]
        out = gt_tensor

        # --- First Degradation ---
        # USM
        if getattr(self.opt, "lq_usm", False):
            out = self.usm_sharpener(out)

        # Blur
        if RNG.get_rng().uniform() < self.opt.blur_prob:
            out = filter2d(out, kernel1)

        # Resize
        updown = random.choices(["up", "down", "keep"], self.opt.resize_prob)[0]
        if updown == "up":
            scale = RNG.get_rng().uniform(1, self.opt.resize_range[1])
        elif updown == "down":
            scale = RNG.get_rng().uniform(self.opt.resize_range[0], 1)
        else:
            scale = 1

        if scale != 1:
            mode = random.choices(
                self.opt.resize_mode_list, weights=self.opt.resize_mode_prob
            )[0]
            out = resize_pt(out, scale_factor=scale, mode=mode)

        # Noise
        if RNG.get_rng().uniform() < self.opt.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt.noise_range,
                clip=True,
                rounds=False,
                gray_prob=self.opt.gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt.poisson_scale_range,
                gray_prob=self.opt.gray_noise_prob,
                clip=True,
                rounds=False,
            )

        # JPEG
        if RNG.get_rng().uniform() < self.opt.jpeg_prob:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt.jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        # Camera Artifacts 1
        out = self._apply_chromatic_aberration(out)
        out = self._apply_demosaicing(out)
        out = self._apply_aliasing(out)

        # --- Second Degradation ---
        # Blur
        if RNG.get_rng().uniform() < self.opt.blur_prob2:
            out = filter2d(out, kernel2)

        # Resize
        updown = random.choices(["up", "down", "keep"], self.opt.resize_prob2)[0]
        if updown == "up":
            scale = RNG.get_rng().uniform(1, self.opt.resize_range2[1])
        elif updown == "down":
            scale = RNG.get_rng().uniform(self.opt.resize_range2[0], 1)
        else:
            scale = 1

        if scale != 1:
            mode = random.choices(
                self.opt.resize_mode_list2, weights=self.opt.resize_mode_prob2
            )[0]
            out = resize_pt(
                out,
                size=(
                    int(ori_h / self.opt.scale * scale),
                    int(ori_w / self.opt.scale * scale),
                ),
                mode=mode,
            )

        # Noise
        if RNG.get_rng().uniform() < self.opt.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt.noise_range2,
                clip=True,
                rounds=False,
                gray_prob=self.opt.gray_noise_prob2,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt.poisson_scale_range2,
                gray_prob=self.opt.gray_noise_prob2,
                clip=True,
                rounds=False,
            )

        # Oversharpening
        out = self._apply_oversharpening(out)

        # JPEG + Sinc + Final Resize
        mode = random.choices(
            self.opt.resize_mode_list3, weights=self.opt.resize_mode_prob3
        )[0]

        if RNG.get_rng().uniform() < 0.5:
            # Resize + Sinc
            out = resize_pt(
                out, size=(ori_h // self.opt.scale, ori_w // self.opt.scale), mode=mode
            )
            out = filter2d(out, sinc_kernel)
            # JPEG
            if RNG.get_rng().uniform() < self.opt.jpeg_prob2:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt.jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG
            if RNG.get_rng().uniform() < self.opt.jpeg_prob2:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt.jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            # Resize + Sinc
            out = resize_pt(
                out, size=(ori_h // self.opt.scale, ori_w // self.opt.scale), mode=mode
            )
            out = filter2d(out, sinc_kernel)

        # --- Paragon OTF ---
        # Modern Compression
        out = ParagonOTF.apply_webp_compression(out, self.opt)
        out = ParagonOTF.apply_avif_compression(out, self.opt)
        out = ParagonOTF.apply_heif_compression(out, self.opt)

        # Camera Artifacts
        out = ParagonOTF.apply_motion_blur(out, self.opt)
        out = ParagonOTF.apply_lens_distortion(out, self.opt)
        out = ParagonOTF.apply_exposure_errors(out, self.opt)
        out = ParagonOTF.apply_color_temperature_shift(out, self.opt)
        out = ParagonOTF.apply_sensor_noise(out, self.opt)
        out = ParagonOTF.apply_rolling_shutter(out, self.opt)

        # Additional
        out = ParagonOTF.apply_oversharpening(out, self.opt)
        out = ParagonOTF.apply_chromatic_aberration(out, self.opt)
        out = ParagonOTF.apply_demosaicing_artifacts(out, self.opt)
        out = ParagonOTF.apply_aliasing_artifacts(out, self.opt)

        return torch.clamp(out, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create OTF-degraded validation dataset from Config"
    )
    parser.add_argument("--input", required=True, help="Input HR validation directory")
    parser.add_argument("--output", required=True, help="Output LR directory")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument(
        "--count", type=int, default=100, help="Number of images to process"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # Setup
    try:
        RNG.init_rng(args.seed)
    except RuntimeError:
        pass  # Already initialized
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Config
    print(f"Loading config: {args.config}")
    config_dict = load_config(args.config)
    opt = ConfigOpt(config_dict)

    # Initialize Pipeline
    print(f"Initializing degradation pipeline on {args.device}...")
    pipeline = DegradationPipeline(opt, device=args.device)

    # Get Images
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_exts])[
        : args.count
    ]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(images)} images...")

    # Process
    for img_path in tqdm(images):
        try:
            # Load
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0
            img_tensor = (
                torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(args.device)
            )

            # Degrade
            with torch.no_grad():
                out_tensor = pipeline.process(img_tensor)

            # Save
            out_img = out_tensor.squeeze(0).cpu().clamp(0, 1).numpy()
            out_img = (out_img * 255).round().astype(np.uint8).transpose(1, 2, 0)

            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"Done. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
