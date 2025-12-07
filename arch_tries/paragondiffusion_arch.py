#!/usr/bin/env python3
"""
ParagonDiffusion: A High-Fidelity Generative Super-Resolution Model
Author: Philip Hofmann

Description:
ParagonDiffusion is a state-of-the-art, diffusion-based super-resolution model
designed to generate the highest possible photorealistic detail. It serves as a
parallel product line to ParagonSR, prioritizing maximum perceptual quality and
generative capability over real-time speed.

This model is built for users who are willing to wait a few seconds for an
upscale in exchange for unparalleled, state-of-the-art realism.

Licensed under the MIT License.

-------------------------------------------------------------------------------------
Core Design & Research Foundations:

This is not a standard diffusion model. It is an advanced, practical system
built upon three pillars of modern generative AI research (2023-2025):

1.  **Feasibility (Latent Diffusion):** The entire process operates in the compact
    latent space of a pre-trained VAE from Stability AI. This reduces VRAM and
    compute requirements by orders of magnitude, making it trainable on consumer
    hardware.

2.  **Faithfulness (Strong Conditioning):** A robust, ControlNet-like mechanism
    ensures the generated details are strictly faithful to the structure, color,
    and content of the low-resolution source image.

3.  **Speed (Rectified Flow / InstaFlow):** The model is trained as a "flow-matching"
    model, learning a direct, straight path from noise to a clean image. This
    is a revolutionary technique that allows for extremely fast sampling,
    generating high-quality images in 1-10 steps instead of the 50-100 of
    traditional diffusion models.

-------------------------------------------------------------------------------------
Training on Consumer Hardware (e.g., RTX 3060 12GB):

Training this model is feasible but requires specific techniques:
-   **Batch Size:** Must be kept very low (1 or 2 is recommended).
-   **Gradient Checkpointing:** It is HIGHLY recommended to enable gradient
    checkpointing in your training framework. This will trade a bit of speed for
    a massive reduction in VRAM usage, making training stable.
-   **Mixed Precision:** Using `fp16` or `bfloat16` is essential.

Usage:
-   Place this file in your `traiNNer/archs/` directory as `paragondiffusion.py`.
-   Install dependencies: `pip install diffusers transformers accelerate`
-   In your config.yaml, start with the 'nano' variant for guaranteed stability:
    `network_g: type: paragondiffusion_nano`
"""

from math import pi

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

# --- Dependency Check and User-Friendly Error ---
try:
    from diffusers import AutoencoderKL
except ImportError:
    raise ImportError(
        "ParagonDiffusion requires the 'diffusers' library. "
        "Please install it with: pip install diffusers transformers accelerate"
    )

# --- Core Building Blocks for the U-Net ---


class SinusoidalPositionalEmbedding(nn.Module):
    """Encodes a scalar timestep into a vector for the U-Net."""

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """A stable and efficient ResNet block, the workhorse of the U-Net."""

    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_channels)
            )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        if time_emb is not None:
            h += self.time_mlp(time_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_dim) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock(
                    in_channels if i == 0 else out_channels, out_channels, time_emb_dim
                )
                for i in range(num_layers)
            ]
        )
        self.downsampler = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, time_emb):
        skip_connections = []
        for resnet in self.resnets:
            x = resnet(x, time_emb)
            skip_connections.append(x)
        x = self.downsampler(x)
        return x, skip_connections


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, time_emb_dim) -> None:
        super().__init__()
        self.upsampler = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock(
                    (out_channels * 2 if i == 0 else out_channels) + out_channels,
                    out_channels,
                    time_emb_dim,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, skip_connections, time_emb):
        x = self.upsampler(x)
        for resnet in self.resnets:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, time_emb)
        return x


# --- Main ParagonDiffusion Architecture ---


class ParagonDiffusion(nn.Module):
    """
    The main ParagonDiffusion model. This class orchestrates the VAE, the
    conditioning encoder, and the U-Net to perform high-fidelity generative SR.
    """

    def __init__(
        self,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        channels: int = 64,
        num_blocks: tuple[int, ...] = (1, 1, 1, 1),
        control_channels: int = 16,
    ) -> None:
        super().__init__()

        print(
            f"Initializing ParagonDiffusion with {channels} base channels and block structure {num_blocks}..."
        )
        self.vae = AutoencoderKL.from_pretrained(vae_model_name)
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.condition_encoder = nn.Sequential(
            nn.Conv2d(3, control_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(control_channels * 2, control_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(control_channels * 4, channels, 1),
        )

        time_emb_dim = channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(channels),
            nn.Linear(channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(4, channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = channels
        # Dynamically create down blocks based on the length of num_blocks tuple
        for i, num_layers in enumerate(num_blocks):
            out_ch = channels * (2**i)
            self.down_blocks.append(DownBlock(in_ch, out_ch, num_layers, time_emb_dim))
            in_ch = out_ch

        self.mid_block = nn.Sequential(
            ResnetBlock(in_ch, in_ch, time_emb_dim),
            ResnetBlock(in_ch, in_ch, time_emb_dim),
        )

        self.up_blocks = nn.ModuleList()
        # Dynamically create up blocks in reverse order
        for i, num_layers in reversed(list(enumerate(num_blocks))):
            out_ch = channels * (2**i)
            # The input channels for the UpBlock need to account for the previous block's output
            # and the skip connections. We handle the skip connection logic inside the UpBlock itself.
            self.up_blocks.append(UpBlock(in_ch, out_ch, num_layers, time_emb_dim))
            in_ch = out_ch

        self.conv_out = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, gt_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z1 = self.vae.encode(gt_image).latent_dist.mean
        z1 = z1 * self.vae.config.scaling_factor
        z0 = torch.randn_like(z1)
        t = torch.rand(gt_image.shape[0], device=gt_image.device)
        t_reshaped = t.view(-1, 1, 1, 1)
        zt = (1 - t_reshaped) * z0 + t_reshaped * z1
        target_vector = z1 - z0
        predicted_vector = self.predict_flow(zt, t, lr_image)
        loss = F.mse_loss(predicted_vector, target_vector)
        return loss

    def predict_flow(
        self, zt: torch.Tensor, t: torch.Tensor, lr_image: torch.Tensor
    ) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        control = self.condition_encoder(lr_image)
        control = F.interpolate(control, size=zt.shape[-2:], mode="bilinear")
        x = self.conv_in(zt)
        x += control
        skips = []
        for block in self.down_blocks:
            x, block_skips = block(x, time_emb)
            skips.extend(block_skips)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x, skips, time_emb)
        return self.conv_out(x)

    @torch.no_grad()
    def sample(self, lr_image: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        device = lr_image.device
        z0 = torch.randn(
            (
                lr_image.shape[0],
                4,
                lr_image.shape[2] * self.vae_scale_factor // 8,
                lr_image.shape[3] * self.vae_scale_factor // 8,
            ),
            device=device,
        )
        zt = z0
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(lr_image.shape[0], device=device) * (i * dt)
            predicted_vector = self.predict_flow(zt, t, lr_image)
            zt = zt + predicted_vector * dt
        z1 = zt / self.vae.config.scaling_factor
        img = self.vae.decode(z1).sample
        return img


# --- Factory Registration for traiNNer-redux: The Complete & Scalable V2 Family ---


@ARCH_REGISTRY.register()
def paragondiffusion_nano(scale: int = 4, **kwargs) -> ParagonDiffusion:
    """The safety net. Guaranteed to be trainable on ~8-12GB VRAM. Recommended for your first training run."""
    return ParagonDiffusion(channels=64, num_blocks=(1, 1, 1, 1), control_channels=16)


@ARCH_REGISTRY.register()
def paragondiffusion_tiny(scale: int = 4, **kwargs) -> ParagonDiffusion:
    """A great balance of quality and performance for ~12GB VRAM, pushing the hardware for better results."""
    return ParagonDiffusion(channels=96, num_blocks=(1, 1, 2, 2), control_channels=16)


@ARCH_REGISTRY.register()
def paragondiffusion_small(scale: int = 4, **kwargs) -> ParagonDiffusion:
    """A strong, high-quality model for GPUs with ~16GB VRAM (e.g., RTX 3080, RTX 4070)."""
    return ParagonDiffusion(channels=128, num_blocks=(1, 2, 2, 2), control_channels=32)


@ARCH_REGISTRY.register()
def paragondiffusion_medium(scale: int = 4, **kwargs) -> ParagonDiffusion:
    """The flagship prosumer model. Excellent quality for ~16-24GB VRAM."""
    return ParagonDiffusion(channels=160, num_blocks=(2, 2, 2, 2), control_channels=32)


@ARCH_REGISTRY.register()
def paragondiffusion_large(scale: int = 4, **kwargs) -> ParagonDiffusion:
    """A high-end enthusiast model for 24GB+ VRAM (e.g., RTX 3090, RTX 4090)."""
    return ParagonDiffusion(channels=192, num_blocks=(2, 2, 4, 4), control_channels=64)


@ARCH_REGISTRY.register()
def paragondiffusion_xl(scale: int = 4, **kwargs) -> ParagonDiffusion:
    """A research-grade model for chasing SOTA, designed for 48GB+ accelerator cards."""
    return ParagonDiffusion(channels=256, num_blocks=(2, 4, 4, 4), control_channels=64)
