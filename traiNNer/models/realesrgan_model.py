import os
import random
import sys
from io import BytesIO
from os import path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import Tensor

try:
    from pillow_avif import AvifImagePlugin
except ImportError:
    AvifImagePlugin = None

from traiNNer.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    resize_pt,
)
from traiNNer.data.transforms import paired_random_crop
from traiNNer.models.sr_model import SRModel
from traiNNer.utils import RNG, DiffJPEG, get_root_logger

# ----------------------------------------------------------------------
# Extended OTF degradations added by Philip Hofmann for ParagonSR
# Degradations included:
#   • WebP compression (webp_prob, webp_range)
#   • AVIF compression (avif_prob, avif_range)
#   • Oversharpening (oversharpen_prob, oversharpen_strength)
#   • Chromatic aberration (chromatic_aberration_prob)
#   • Demosaicing artifacts (demosaic_prob)
#   • Aliasing artifacts (aliasing_prob, aliasing_scale_range)
# ----------------------------------------------------------------------
from traiNNer.utils.img_process_util import USMSharp, filter2d
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.registry import MODEL_REGISTRY
from traiNNer.utils.types import DataFeed

from .paragon_otf_degradations import ParagonOTF

OTF_DEBUG_PATH = osp.abspath(
    osp.abspath(osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/otf"))
)

ANTIALIAS_MODES = {"bicubic", "bilinear"}


@MODEL_REGISTRY.register(suffix="traiNNer")
class RealESRGANModel(SRModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        # Initialize sequence controller if enabled
        self.sequence_controller: SequenceController | None = None
        if hasattr(opt, "enable_sequences") and opt.enable_sequences:
            sequences = create_enhanced_predefined_sequences()
            self.sequence_controller = SequenceController(sequences)
            logger = get_root_logger()
            logger.info(
                "ParagonSR sequence control enabled with %d predefined sequences.",
                len(sequences),
            )

        self.queue_lr: Tensor | None = None
        self.queue_gt: Tensor | None = None
        self.queue_ptr = 0
        self.kernel1: Tensor | None = None
        self.kernel2: Tensor | None = None
        self.sinc_kernel: Tensor | None = None

        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.queue_size = opt.queue_size

        self.otf_debug = opt.high_order_degradations_debug
        self.otf_debug_limit = opt.high_order_degradations_debug_limit

        if self.otf_debug:
            logger = get_root_logger()
            logger.info(
                "OTF debugging enabled. LR tiles will be saved to: %s", OTF_DEBUG_PATH
            )

    def _apply_webp_compression(self, img_tensor: Tensor) -> Tensor:
        """Apply WebP compression to a batch of images.

        Args:
            img_tensor (Tensor): Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Tensor: WebP compressed tensor of the same shape.
        """
        if not hasattr(self.opt, "webp_prob") or not hasattr(self.opt, "webp_range"):
            return img_tensor

        if RNG.get_rng().uniform() >= self.opt.webp_prob:
            return img_tensor

        # Convert tensor to PIL Images, apply WebP compression, and convert back
        batch_size = img_tensor.size(0)
        compressed_images = []

        # Get quality parameter
        quality = RNG.get_rng().uniform(self.opt.webp_range[0], self.opt.webp_range[1])

        for i in range(batch_size):
            # Convert tensor to PIL Image
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8")
            img = img.transpose(1, 2, 0)  # CHW to HWC
            if img.shape[2] == 1:
                img = img[:, :, 0]  # Convert to grayscale if needed
            pil_img = Image.fromarray(img)

            # Apply WebP compression
            buffer = BytesIO()
            pil_img.save(buffer, format="WEBP", quality=int(quality))
            buffer.seek(0)

            # Load compressed image
            compressed_img = Image.open(buffer)
            compressed_img = compressed_img.convert("RGB")

            # Convert back to tensor
            compressed_tensor = (
                torch.from_numpy(np.array(compressed_img)).float() / 255.0
            )
            compressed_tensor = compressed_tensor.permute(2, 0, 1)  # HWC to CHW
            compressed_images.append(compressed_tensor)

        return torch.stack(compressed_images, dim=0).to(img_tensor.device)

    def _apply_avif_compression(self, img_tensor: Tensor) -> Tensor:
        """Apply AVIF compression to a batch of images.

        Args:
            img_tensor (Tensor): Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Tensor: AVIF compressed tensor of the same shape.
        """
        if not hasattr(self.opt, "avif_prob") or not hasattr(self.opt, "avif_range"):
            return img_tensor

        if RNG.get_rng().uniform() >= self.opt.avif_prob:
            return img_tensor

        # Convert tensor to PIL Images, apply AVIF compression, and convert back
        batch_size = img_tensor.size(0)
        compressed_images = []

        # Get quality parameter
        quality = RNG.get_rng().uniform(self.opt.avif_range[0], self.opt.avif_range[1])

        for i in range(batch_size):
            # Convert tensor to PIL Image
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8")
            img = img.transpose(1, 2, 0)  # CHW to HWC
            if img.shape[2] == 1:
                img = img[:, :, 0]  # Convert to grayscale if needed
            pil_img = Image.fromarray(img)

            # Apply AVIF compression
            buffer = BytesIO()
            pil_img.save(buffer, format="AVIF", quality=int(quality))
            buffer.seek(0)

            # Load compressed image
            compressed_img = Image.open(buffer)
            compressed_img = compressed_img.convert("RGB")

            # Convert back to tensor
            compressed_tensor = (
                torch.from_numpy(np.array(compressed_img)).float() / 255.0
            )
            compressed_tensor = compressed_tensor.permute(2, 0, 1)  # HWC to CHW
            compressed_images.append(compressed_tensor)

        return torch.stack(compressed_images, dim=0).to(img_tensor.device)

    def _apply_oversharpening(self, img_tensor: Tensor) -> Tensor:
        """Apply oversharpening artifact simulation to a batch of images.

        Args:
            img_tensor (Tensor): Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Tensor: Oversharpened tensor of the same shape.
        """
        if not hasattr(self.opt, "oversharpen_prob") or not hasattr(
            self.opt, "oversharpen_strength"
        ):
            return img_tensor

        if RNG.get_rng().uniform() >= self.opt.oversharpen_prob:
            return img_tensor

        # Get strength parameter
        strength = RNG.get_rng().uniform(
            self.opt.oversharpen_strength[0], self.opt.oversharpen_strength[1]
        )

        # Create a blurred copy of the image using Gaussian blur
        from traiNNer.data.degradations import resize_pt

        blurred = resize_pt(
            img_tensor, mode="bicubic", scale_factor=1.0
        )  # This is just a placeholder

        # For a proper Gaussian blur, we would need to implement it or use a library
        # For now, we'll use a simple box blur as an approximation
        kernel_size = 5
        padding = kernel_size // 2
        weight = torch.ones(
            1, 1, kernel_size, kernel_size, device=img_tensor.device
        ) / (kernel_size**2)
        weight = weight.repeat(img_tensor.size(1), 1, 1, 1)

        blurred = torch.nn.functional.conv2d(
            img_tensor, weight, padding=padding, groups=img_tensor.size(1)
        )

        # Subtract the blurred copy from the original to isolate high-frequency details
        details = img_tensor - blurred

        # Add a multiplied version of these details back to the original image
        oversharpened = img_tensor + details * strength

        # Clamp values to [0, 1]
        return torch.clamp(oversharpened, 0, 1)

    def _apply_chromatic_aberration(self, img_tensor: Tensor) -> Tensor:
        """Apply chromatic aberration simulation to a batch of images.

        Args:
            img_tensor (Tensor): Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Tensor: Chromatic aberration applied tensor of the same shape.
        """
        if not hasattr(self.opt, "chromatic_aberration_prob"):
            return img_tensor

        if RNG.get_rng().uniform() >= self.opt.chromatic_aberration_prob:
            return img_tensor

        batch_size, channels, _height, _width = img_tensor.shape
        if channels != 3:
            # Only apply to RGB images
            return img_tensor

        # Split the image tensor's R, G, B channels
        r_channel, g_channel, b_channel = (
            img_tensor[:, 0:1, :, :],
            img_tensor[:, 1:2, :, :],
            img_tensor[:, 2:3, :, :],
        )

        # Create slight scaling factors for R and B channels
        # R at 100.1%, B at 99.9%
        scale_r = 1.001
        scale_b = 0.999

        # Create affine transformation matrices for scaling
        # For R channel (scale up)
        theta_r = torch.tensor(
            [[[scale_r, 0, 0], [0, scale_r, 0]]],
            dtype=torch.float32,
            device=img_tensor.device,
        )
        theta_r = theta_r.repeat(batch_size, 1, 1)
        grid_r = torch.nn.functional.affine_grid(
            theta_r, r_channel.size(), align_corners=False
        )
        r_channel_scaled = torch.nn.functional.grid_sample(
            r_channel, grid_r, align_corners=False, mode="bilinear"
        )

        # For B channel (scale down)
        theta_b = torch.tensor(
            [[[scale_b, 0, 0], [0, scale_b, 0]]],
            dtype=torch.float32,
            device=img_tensor.device,
        )
        theta_b = theta_b.repeat(batch_size, 1, 1)
        grid_b = torch.nn.functional.affine_grid(
            theta_b, b_channel.size(), align_corners=False
        )
        b_channel_scaled = torch.nn.functional.grid_sample(
            b_channel, grid_b, align_corners=False, mode="bilinear"
        )

        # G channel remains untouched
        # Merge the channels back together
        result = torch.cat([r_channel_scaled, g_channel, b_channel_scaled], dim=1)

        # Clamp values to [0, 1]
        return torch.clamp(result, 0, 1)

    def _apply_demosaicing_artifacts(self, img_tensor: Tensor) -> Tensor:
        """Apply demosaicing artifact simulation to a batch of images.

        Args:
            img_tensor (Tensor): Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Tensor: Demosaicing artifact applied tensor of the same shape.
        """
        if not hasattr(self.opt, "demosaic_prob"):
            return img_tensor

        if RNG.get_rng().uniform() >= self.opt.demosaic_prob:
            return img_tensor

        # Convert tensor to numpy for OpenCV processing
        batch_size = img_tensor.size(0)
        processed_images = []

        for i in range(batch_size):
            # Convert tensor to numpy array
            img = img_tensor[i].cpu().clamp(0, 1).numpy()
            img = (img * 255).astype("uint8")
            img = img.transpose(1, 2, 0)  # CHW to HWC

            # Convert RGB to Bayer pattern (BGGR)
            h, w = img.shape[:2]
            bayer_img = np.zeros((h, w), dtype=np.uint8)

            # Create BGGR Bayer pattern
            bayer_img[0::2, 0::2] = img[0::2, 0::2, 2]  # Red
            bayer_img[0::2, 1::2] = img[0::2, 1::2, 1]  # Green
            bayer_img[1::2, 0::2] = img[1::2, 0::2, 1]  # Green
            bayer_img[1::2, 1::2] = img[1::2, 1::2, 0]  # Blue

            # Apply demosaicing using OpenCV
            try:
                import cv2

                demosaiced_img = cv2.demosaicing(bayer_img, cv2.COLOR_BAYER_BG2BGR)
                # Convert back to tensor
                demosaiced_tensor = torch.from_numpy(demosaiced_img).float() / 255.0
                demosaiced_tensor = demosaiced_tensor.permute(2, 0, 1)  # HWC to CHW
                processed_images.append(demosaiced_tensor)
            except ImportError:
                # If OpenCV is not available, return original image
                processed_images.append(img_tensor[i])

        if processed_images:
            return torch.stack(processed_images, dim=0).to(img_tensor.device)
        else:
            return img_tensor

    def _apply_aliasing_artifacts(self, img_tensor: Tensor) -> Tensor:
        """Apply aliasing artifact simulation to a batch of images.

        Args:
            img_tensor (Tensor): Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Tensor: Aliasing artifact applied tensor of the same shape.
        """
        if not hasattr(self.opt, "aliasing_prob") or not hasattr(
            self.opt, "aliasing_scale_range"
        ):
            return img_tensor

        if RNG.get_rng().uniform() >= self.opt.aliasing_prob:
            return img_tensor

        # Get scale factor
        scale_factor = RNG.get_rng().uniform(
            self.opt.aliasing_scale_range[0], self.opt.aliasing_scale_range[1]
        )

        # Perform a downscale-then-upscale cycle using nearest-neighbor interpolation
        _batch_size, _channels, height, width = img_tensor.shape

        # Downscale
        downscale_size = (int(height * scale_factor), int(width * scale_factor))
        downscaled = torch.nn.functional.interpolate(
            img_tensor, size=downscale_size, mode="nearest"
        )

        # Upscale back to original size
        upscaled = torch.nn.functional.interpolate(
            downscaled, size=(height, width), mode="nearest"
        )

        return upscaled

    @torch.no_grad()
    def _dequeue_and_enqueue(self) -> None:
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """

        assert self.lq is not None
        assert self.gt is not None

        # initialize
        b, c, h, w = self.lq.size()
        if self.queue_lr is None:
            assert self.queue_size % b == 0, (
                f"queue size {self.queue_size} should be divisible by batch size {b}"
            )
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            assert self.queue_lr is not None
            assert self.queue_gt is not None
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            assert self.queue_lr is not None
            assert self.queue_gt is not None

            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data: DataFeed) -> None:
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        if self.is_train:
            assert (
                "gt" in data
                and "kernel1" in data
                and "kernel2" in data
                and "sinc_kernel" in data
            )
            # training data synthesis
            self.gt = data["gt"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            self.kernel1 = data["kernel1"].to(
                self.device,
                non_blocking=True,
            )
            self.kernel2 = data["kernel2"].to(
                self.device,
                non_blocking=True,
            )
            self.sinc_kernel = data["sinc_kernel"].to(
                self.device,
                non_blocking=True,
            )

            ori_h, ori_w = self.gt.size()[2:4]

            # Clean Pass-Through: If random check passes, return original GT as LQ and skip all degradations
            if (
                hasattr(self.opt, "p_clean")
                and RNG.get_rng().uniform() < self.opt.p_clean
            ):
                self.lq = self.gt.clone()
                # Clamp and round
                self.lq = torch.clamp((self.lq * 255.0).round(), 0, 255) / 255.0
                # Random crop
                gt_size = self.opt.datasets["train"].gt_size
                assert gt_size is not None
                self.gt, self.lq = paired_random_crop(
                    self.gt, self.lq, gt_size, self.opt.scale
                )
                # Training pair pool
                self._dequeue_and_enqueue()
                self.lq = self.lq.contiguous()
                return

            # ========================================================================
            # PHYSICALLY ACCURATE DEGRADATION ORDER
            # Matches validation script: Optics → Sensor → ISP → Compression
            # ========================================================================

            # ========================================================================
            # STAGE 1: CAMERA OPTICS (light entering camera)
            # Applied before sensor capture
            # ========================================================================

            # Lens distortion (barrel/pincushion from lens geometry)
            out = ParagonOTF.apply_lens_distortion(self.gt, self.opt)

            # Chromatic aberration (color fringing from lens dispersion)
            out = self._apply_chromatic_aberration(out)

            # Motion blur (camera shake or subject movement)
            out = ParagonOTF.apply_motion_blur(out, self.opt)

            # Defocus blur (depth of field, lens blur)
            if RNG.get_rng().uniform() < self.opt.blur_prob:
                out = filter2d(out, self.kernel1)

            # ========================================================================
            # STAGE 2: SENSOR CAPTURE (light → digital signal)
            # Physical processes as photons are converted to electrons
            # ========================================================================

            # Demosaicing (Bayer pattern → RGB conversion)
            out = self._apply_demosaicing_artifacts(out)

            # Sensor noise (captured AFTER optics, so noise is never blurred)
            out = ParagonOTF.apply_sensor_noise(out, self.opt)

            # Rolling shutter (CMOS sensor artifacts)
            out = ParagonOTF.apply_rolling_shutter(out, self.opt)

            # ========================================================================
            # STAGE 3: CAMERA ISP PROCESSING (in-camera processing)
            # Digital processing pipeline before saving
            # ========================================================================

            # Exposure adjustment
            out = ParagonOTF.apply_exposure_errors(out, self.opt)

            # White balance / color temperature correction
            out = ParagonOTF.apply_color_temperature_shift(out, self.opt)

            # Camera sharpening (often oversharpening)
            out = self._apply_oversharpening(out)

            # In-camera downsampling (creates aliasing artifacts)
            out = self._apply_aliasing_artifacts(out)

            # Final resize to output resolution
            assert len(self.opt.resize_mode_list3) == len(self.opt.resize_mode_prob3), (
                "resize_mode_list3 and resize_mode_prob3 must be the same length"
            )

            mode = random.choices(
                self.opt.resize_mode_list3, weights=self.opt.resize_mode_prob3
            )[0]
            out = resize_pt(
                out,
                size=(ori_h // self.opt.scale, ori_w // self.opt.scale),
                mode=mode,
            )

            # Apply sinc filter for anti-aliasing
            out = filter2d(out, self.sinc_kernel)

            # ========================================================================
            # STAGE 4: INITIAL COMPRESSION (camera saves file)
            # Use unified compression pipeline for realistic format selection
            # ========================================================================

            out = ParagonOTF.apply_realistic_compression_pipeline(out, self.opt)

            # ========================================================================
            # STAGE 5: OPTIONAL EDITING (post-processing before upload)
            # Social media filters and editing apps
            # ========================================================================

            editing_prob = self.opt.get("editing_prob", 0.0)
            if RNG.get_rng().uniform() < editing_prob:
                # Additional exposure tweaks
                if hasattr(self.opt, "editing_exposure_prob"):
                    if RNG.get_rng().uniform() < self.opt.editing_exposure_prob:
                        out = ParagonOTF.apply_exposure_errors(out, self.opt)

                # Additional sharpening
                if hasattr(self.opt, "editing_oversharpen_prob"):
                    if RNG.get_rng().uniform() < self.opt.editing_oversharpen_prob:
                        out = ParagonOTF.apply_oversharpening(out, self.opt)

            # ========================================================================
            # NOTE: STAGE 6 (platform recompression) is handled inside
            # apply_realistic_compression_pipeline() via recompression_prob
            # ========================================================================

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # random crop
            gt_size = self.opt.datasets["train"].gt_size
            assert gt_size is not None
            self.gt, self.lq = paired_random_crop(
                self.gt, self.lq, gt_size, self.opt.scale
            )

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            i = 1
            if self.otf_debug:
                os.makedirs(OTF_DEBUG_PATH, exist_ok=True)
                while os.path.exists(rf"{OTF_DEBUG_PATH}/{i:06d}_otf_lq.png"):
                    i += 1

                if i <= self.otf_debug_limit or self.otf_debug_limit == 0:
                    torchvision.utils.save_image(
                        self.lq,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_lq.png"),
                        padding=0,
                    )

                    torchvision.utils.save_image(
                        self.gt,
                        os.path.join(OTF_DEBUG_PATH, f"{i:06d}_otf_gt.png"),
                        padding=0,
                    )

            # moa
            if self.is_train and self.batch_augment:
                self.gt, self.lq = self.batch_augment(self.gt, self.lq)
        else:
            # for paired training or validation
            assert "lq" in data
            self.lq = data["lq"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            if "gt" in data:
                self.gt = data["gt"].to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )
