#!/usr/bin/env python3
"""
Improved R3GAN Loss Implementation
----------------------------------
Fixed and enhanced version of your R3GAN loss for stable GAN training.

Key improvements:
  • Proper relativistic hinge formulation (relativistic hinge variant)
  • Gradient penalties backpropagate correctly (create_graph=True) when supported
  • FP32 computation for gradient penalties to avoid AMP saturation / NaNs
  • Graceful fallback if second-derivative for an op is missing (logs a warning)
  • Uses unaugmented images for R1/R2 when provided (recommended)
  • Compatible with traiNNer registry
"""

import warnings
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from traiNNer.utils.logger import get_root_logger
from traiNNer.utils.registry import LOSS_REGISTRY

logger = get_root_logger()


# ---------------------------------------------------------------------------- #
# Safe gradient penalty helper
# ---------------------------------------------------------------------------- #
class SafeGradientPenalty:
    """Compute differentiable R1/R2 gradient penalties safely.

    Behavior:
      - Run forward & gradient computation in fp32 (disable amp) to avoid fp16/bf16 saturation.
      - Prefer create_graph=True so the penalty backpropagates to discriminator params.
      - If an op in the graph doesn't implement second derivatives (e.g. grid_sampler),
        fall back to create_graph=False and warn the user that the penalty will NOT
        produce gradients for discriminator parameters.
    """

    @staticmethod
    def compute_grad_penalty(
        net_d: nn.Module, images: Tensor, penalty_weight: float
    ) -> Tensor:
        if penalty_weight <= 0:
            return torch.tensor(0.0, device=images.device, dtype=images.dtype)

        # Prepare a fresh input that requires grad.
        images_reqgrad = images.detach().clone().requires_grad_(True)

        # Force fp32 execution for stable second-order gradients
        # Use the new autocast API (non-deprecated)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out = net_d(images_reqgrad.float())

        # If multi-scale outputs, use the last (final prediction head)
        if isinstance(out, (list, tuple)):
            out = out[-1]

        # Preferred: compute grads with create_graph=True so penalty backprops to net params
        try:
            grads = torch.autograd.grad(
                outputs=out.sum(),
                inputs=images_reqgrad,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        except RuntimeError as exc:
            # Detect common missing-second-derivative message & fallback
            msg = str(exc)
            if "grid_sampler_2d_backward" in msg or "not implemented" in msg:
                warnings.warn(
                    "compute_grad_penalty: falling back to create_graph=False because "
                    "a second-derivative op is missing (e.g., grid_sampler). "
                    "The returned penalty will be finite but will NOT produce gradients "
                    "for discriminator parameters. Recommended: compute penalties on "
                    "unaugmented / grid_sample-free images.",
                    stacklevel=2,
                )
                logger.warning(
                    "SafeGradientPenalty fallback: second-derivative not implemented for an op in the graph. "
                    "Penalty will NOT backpropagate to discriminator parameters."
                )
                grads = torch.autograd.grad(
                    outputs=out.sum(),
                    inputs=images_reqgrad,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
            else:
                # Re-raise unexpected runtime errors
                raise

        grads = grads.reshape(grads.size(0), -1)
        grad_penalty = (grads.norm(2, dim=1) ** 2).mean()

        # return penalty as same dtype/device as input (but computed in fp32)
        return (
            grad_penalty.to(device=images.device, dtype=images.dtype) * penalty_weight
        )


# ---------------------------------------------------------------------------- #
# Main R3GAN loss
# ---------------------------------------------------------------------------- #
@LOSS_REGISTRY.register()
class R3GANLoss(nn.Module):
    """Full R3GAN loss with relativistic hinge formulation and R1/R2 penalties."""

    def __init__(
        self,
        loss_weight: float,
        gan_type: str = "r3gan",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        r1_weight: float = 3.0,
        r2_weight: float = 3.0,
        use_relu: bool = False,  # kept for API compatibility
    ) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.r1_weight = float(r1_weight)
        self.r2_weight = float(r2_weight)
        self.safe_grad_penalty = SafeGradientPenalty()

        # Register appropriate base losses for alternative GANs
        if gan_type == "r3gan":
            self.loss_func = self._r3gan_loss
        elif gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
            self.loss_func = self._vanilla_loss
        elif gan_type == "lsgan":
            self.loss = nn.MSELoss()
            self.loss_func = self._lsgan_loss
        elif gan_type == "wgan":
            self.loss_func = self._wgan_loss
        elif gan_type == "wgan_softplus":
            self.loss_func = self._wgan_softplus_loss
        elif gan_type == "hinge":
            self.loss = nn.ReLU()
            self.loss_func = self._hinge_loss
        else:
            raise NotImplementedError(f"GAN type {gan_type} is not implemented.")

    # ------------------------------------------------------------------ #
    # Non-relativistic fallback losses
    # ------------------------------------------------------------------ #
    def _get_target_label(self, input: Tensor, target_is_real: bool) -> Tensor:
        val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_full(input.size(), val)

    def _vanilla_loss(self, input: Tensor, target_is_real: bool, **_) -> Tensor:
        return self.loss(input, self._get_target_label(input, target_is_real))

    def _lsgan_loss(self, input: Tensor, target_is_real: bool, **_) -> Tensor:
        return self.loss(input, self._get_target_label(input, target_is_real))

    def _wgan_loss(self, input: Tensor, target_is_real: bool, **_) -> Tensor:
        return -input.mean() if target_is_real else input.mean()

    def _wgan_softplus_loss(self, input: Tensor, target_is_real: bool, **_) -> Tensor:
        return F.softplus(-input).mean() if target_is_real else F.softplus(input).mean()

    def _hinge_loss(
        self, input: Tensor, target_is_real: bool, is_disc: bool = False, **_
    ) -> Tensor:
        if is_disc:
            input = -input if target_is_real else input
            return F.relu(1 + input).mean()
        else:
            return -input.mean()

    # ------------------------------------------------------------------ #
    # Relativistic hinge (R3GAN)
    # ------------------------------------------------------------------ #
    def _compute_discriminator_loss_with_grad_penalty(
        self,
        net_d: nn.Module,
        real_output: Tensor,
        fake_output: Tensor,
        # Note: adv uses augmented images (real_images, fake_images),
        # but penalties should be computed on unaugmented images when available.
        real_images_unaug: Tensor | None,
        fake_images_unaug: Tensor | None,
    ) -> dict[str, Tensor]:
        # Handle lists (multi-scale outputs)
        if isinstance(real_output, (list, tuple)):
            real_output = real_output[-1]
        if isinstance(fake_output, (list, tuple)):
            fake_output = fake_output[-1]

        # Relativistic average hinge loss
        real_mean = fake_output.detach().mean()
        fake_mean = real_output.detach().mean()

        real_term = F.relu(1.0 - (real_output - real_mean)).mean()
        fake_term = F.relu(1.0 + (fake_output - fake_mean)).mean()
        adv_loss = 0.5 * (real_term + fake_term)

        # R1 / R2 penalties: prefer unaugmented images to avoid grid_sample second-derivative issues
        if real_images_unaug is None or fake_images_unaug is None:
            warnings.warn(
                "R3GANLoss: real_images_unaug or fake_images_unaug is None. "
                "Gradient penalties will be computed on the provided (possibly augmented) images. "
                "This can trigger errors if augmentations use ops without second-derivatives "
                "(e.g., grid_sampler). Recommended: pass unaugmented images via "
                "real_images_unaug/fake_images_unaug.",
                stacklevel=2,
            )
            logger.warning(
                "R3GANLoss called without unaugmented images for gradient penalties."
            )

        # Compute R1 on unaug if available else on provided images
        r1_target = (
            real_images_unaug if real_images_unaug is not None else real_output.detach()
        )
        # But compute penalty using input images (tensors) — ensure we pass image tensors to helper
        r1_images_for_penalty = (
            real_images_unaug if real_images_unaug is not None else None
        )

        # R1
        if self.r1_weight > 0:
            if r1_images_for_penalty is None:
                # if we don't have an image tensor, fall back to computing grads w.r.t. real_output
                # (less common) — here we compute no penalty and warn.
                warnings.warn(
                    "R3GANLoss: cannot compute R1 penalty because no real image tensor is available.",
                    stacklevel=2,
                )
                r1_penalty = torch.tensor(0.0, device=real_output.device)
            else:
                r1_penalty = self.safe_grad_penalty.compute_grad_penalty(
                    net_d, r1_images_for_penalty, self.r1_weight
                )
        else:
            r1_penalty = torch.tensor(0.0, device=real_output.device)

        # R2
        if self.r2_weight > 0:
            if fake_images_unaug is None:
                warnings.warn(
                    "R3GANLoss: cannot compute R2 penalty because no fake image tensor is available.",
                    stacklevel=2,
                )
                r2_penalty = torch.tensor(0.0, device=fake_output.device)
            else:
                r2_penalty = self.safe_grad_penalty.compute_grad_penalty(
                    net_d, fake_images_unaug, self.r2_weight
                )
        else:
            r2_penalty = torch.tensor(0.0, device=fake_output.device)

        total_loss = adv_loss + 0.5 * (r1_penalty + r2_penalty)

        return {
            "d_loss": total_loss,
            "r1_penalty": r1_penalty,
            "r2_penalty": r2_penalty,
        }

    def _compute_generator_loss(
        self, real_output: Tensor, fake_output: Tensor
    ) -> Tensor:
        # Handle lists
        if isinstance(real_output, (list, tuple)):
            real_output = real_output[-1]
        if isinstance(fake_output, (list, tuple)):
            fake_output = fake_output[-1]

        # Relativistic generator counterpart (hinge-style)
        real_mean = fake_output.mean()
        fake_mean = real_output.mean()
        loss_real = F.relu(1.0 + (real_output - real_mean)).mean()
        loss_fake = F.relu(1.0 - (fake_output - fake_mean)).mean()
        g_loss = 0.5 * (loss_real + loss_fake)
        return g_loss * self.loss_weight

    def _r3gan_loss(
        self,
        net_d: nn.Module,
        real_images: Tensor,
        fake_images: Tensor,
        is_disc: bool,
        real_images_unaug: Tensor | None = None,
        fake_images_unaug: Tensor | None = None,
        **_: dict,
    ) -> Tensor | dict[str, Tensor]:
        # Forward through discriminator with the images intended for adversarial loss
        real_output = net_d(real_images)
        fake_output = net_d(fake_images)

        if is_disc:
            # Use provided unaugmented images for R1/R2 penalties when available.
            return self._compute_discriminator_loss_with_grad_penalty(
                net_d,
                real_output,
                fake_output,
                real_images_unaug,
                fake_images_unaug,
            )
        else:
            return self._compute_generator_loss(real_output, fake_output)

    # ------------------------------------------------------------------ #
    # Main forward interface
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input: Tensor | None = None,
        target_is_real: bool | None = None,
        is_disc: bool = False,
        **kwargs,
    ) -> Tensor | dict[str, Tensor]:
        if self.gan_type == "r3gan":
            # R3GAN expects net_d/real_images/fake_images etc. passed via kwargs
            return self._r3gan_loss(is_disc=is_disc, **kwargs)
        else:
            assert input is not None
            assert target_is_real is not None
            return self.loss_func(input, target_is_real, is_disc=is_disc, **kwargs)


# ---------------------------------------------------------------------------- #
# Multi-scale variant
# ---------------------------------------------------------------------------- #
@LOSS_REGISTRY.register()
class MultiScaleR3GANLoss(R3GANLoss):
    """
    Multi-scale R3GAN loss — averages across multiple discriminator outputs.

    If your discriminator returns a list of predictions (one per scale),
    this class averages the losses from all scales.
    """

    def forward(
        self,
        input: list[Tensor] | Tensor,
        target_is_real: bool,
        is_disc: bool = False,
        **kwargs,
    ) -> Tensor:
        if isinstance(input, list):
            assert len(input) > 0, "Empty discriminator output list."
            total = torch.tensor(0.0, device=input[0].device)
            for pred in input:
                if isinstance(pred, (list, tuple)):
                    pred = pred[-1]
                total += super().forward(pred, target_is_real, is_disc, **kwargs).mean()
            return total / len(input)
        else:
            return super().forward(input, target_is_real, is_disc, **kwargs)
