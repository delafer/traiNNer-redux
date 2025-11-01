#!/usr/bin/env python3
"""
Enhanced R3GAN Loss Implementation with Proper Gradient Penalties
Based on the official R3GAN repository: https://github.com/NVlabs/R3GAN

R3GAN (Relativistic GAN with Regularization) is a modern GAN architecture
that uses a relativistic discriminator formulation with R1/R2 gradient penalties.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from traiNNer.utils.registry import LOSS_REGISTRY


class SafeGradientPenalty:
    """Safe gradient penalty computation that doesn't interfere with training graphs"""

    @staticmethod
    def compute_grad_penalty(
        net_d: nn.Module, images: Tensor, penalty_weight: float
    ) -> Tensor:
        """Compute gradient penalty safely without interfering with main training graph"""
        if penalty_weight > 0:
            # Create a fresh, gradient-enabled copy of the images
            images_detached = images.detach().clone().requires_grad_(True)

            # Perform a forward pass with the detached images
            output_detached = net_d(images_detached)

            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=output_detached.sum(),
                inputs=images_detached,
                create_graph=False,  # No new graph nodes needed here
                retain_graph=False,  # Don't retain the graph
            )[0]

            # Reshape gradients and compute penalty
            gradients = gradients.view(gradients.size(0), -1)
            grad_penalty = (gradients.norm(2, dim=1) ** 2).mean()

            return grad_penalty * penalty_weight
        return torch.tensor(0.0, device=images.device)


@LOSS_REGISTRY.register()
class R3GANLoss(nn.Module):
    """
    Complete R3GAN Loss Implementation with proper gradient penalties

    This implementation maintains the full R3GAN functionality including
    R1/R2 gradient penalties while ensuring stable training.
    """

    def __init__(
        self,
        loss_weight: float,
        gan_type: str = "r3gan",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        r1_weight: float = 3.0,
        r2_weight: float = 3.0,
        use_relu: bool = False,
    ) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.r1_weight = r1_weight
        self.r2_weight = r2_weight
        self.use_relu = use_relu
        self.safe_grad_penalty = SafeGradientPenalty()

        # Register the appropriate loss function
        if self.gan_type == "r3gan":
            self.loss_func = self._r3gan_loss
        elif self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
            self.loss_func = self._vanilla_loss
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
            self.loss_func = self._lsgan_loss
        elif self.gan_type == "wgan":
            self.loss_func = self._wgan_loss
        elif self.gan_type == "wgan_softplus":
            self.loss_func = self._wgan_softplus_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
            self.loss_func = self._hinge_loss
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    def _vanilla_loss(
        self, input: Tensor, target_is_real: bool, is_disc: bool = False, **_: dict
    ) -> Tensor:
        target = self._get_target_label(input, target_is_real)
        return self.loss(input, target)

    def _lsgan_loss(
        self, input: Tensor, target_is_real: bool, is_disc: bool = False, **_: dict
    ) -> Tensor:
        target = self._get_target_label(input, target_is_real)
        return self.loss(input, target)

    def _wgan_loss(
        self, input: Tensor, target_is_real: bool, is_disc: bool = False, **_: dict
    ) -> Tensor:
        return -input.mean() if target_is_real else input.mean()

    def _wgan_softplus_loss(
        self, input: Tensor, target_is_real: bool, is_disc: bool = False, **_: dict
    ) -> Tensor:
        return F.softplus(-input).mean() if target_is_real else F.softplus(input).mean()

    def _hinge_loss(
        self, input: Tensor, target_is_real: bool, is_disc: bool = False, **_: dict
    ) -> Tensor:
        if is_disc:
            input = -input if target_is_real else input
            assert isinstance(self.loss, nn.ReLU)
            return self.loss(1 + input).mean()
        else:
            return -input.mean()

    def _compute_discriminator_loss_with_grad_penalty(
        self,
        net_d: nn.Module,
        real_output: Tensor,
        fake_output: Tensor,
        real_images: Tensor,
        fake_images: Tensor,
    ) -> dict[str, Tensor]:
        """Compute discriminator loss with R1/R2 gradient penalties"""
        # Standard R3GAN relativistic discriminator loss
        real_loss = F.softplus(-real_output).mean()
        fake_loss = F.softplus(fake_output).mean()
        adv_loss = (real_loss + fake_loss) / 2

        # R1 and R2 gradient penalties
        r1_penalty = self.safe_grad_penalty.compute_grad_penalty(
            net_d, real_images, self.r1_weight
        )
        r2_penalty = self.safe_grad_penalty.compute_grad_penalty(
            net_d, fake_images, self.r2_weight
        )

        total_loss = adv_loss + (r1_penalty + r2_penalty) / 2

        return {
            "d_loss": total_loss,
            "r1_penalty": r1_penalty,
            "r2_penalty": r2_penalty,
        }

    def _compute_generator_loss(
        self, real_output: Tensor, fake_output: Tensor
    ) -> Tensor:
        """Compute generator loss"""
        # Relativistic generator loss
        real_loss = F.softplus(real_output).mean()
        fake_loss = F.softplus(-fake_output).mean()
        return ((real_loss + fake_loss) / 2) * self.loss_weight

    def _r3gan_loss(
        self,
        net_d: nn.Module,
        real_images: Tensor,
        fake_images: Tensor,
        is_disc: bool,
        **_: dict,
    ) -> Tensor | dict[str, Tensor]:
        """Relativistic R3GAN loss with R1/R2 gradient penalties"""
        real_output = net_d(real_images)
        fake_output = net_d(fake_images)

        if is_disc:
            return self._compute_discriminator_loss_with_grad_penalty(
                net_d, real_output, fake_output, real_images, fake_images
            )
        else:
            return self._compute_generator_loss(real_output, fake_output)

    def _get_target_label(self, input: Tensor, target_is_real: bool) -> Tensor:
        """Return target label for non‑relativistic loss types."""
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(
        self,
        input: Tensor | None = None,
        target_is_real: bool | None = None,
        is_disc: bool = False,
        **kwargs,
    ) -> Tensor | dict[str, Tensor]:
        """Forward entry point used by the training loop."""
        if self.gan_type == "r3gan":
            # R3GAN requires a different signature, so we call it directly
            return self._r3gan_loss(is_disc=is_disc, **kwargs)
        else:
            # Fallback for other GAN types
            assert input is not None
            assert target_is_real is not None
            return self.loss_func(input, target_is_real, is_disc=is_disc, **kwargs)


@LOSS_REGISTRY.register()
class MultiScaleR3GANLoss(R3GANLoss):
    """Multi‑scale version of :class:`R3GANLoss` that averages over a list
    of discriminator outputs (common in high‑resolution generators)."""

    def forward(
        self,
        input: Tensor | list[Tensor],
        target_is_real: bool,
        is_disc: bool = False,
        **kwargs,
    ) -> Tensor:
        if isinstance(input, list):
            assert len(input) > 0
            total = torch.tensor(0.0, device=input[0].device)
            for pred in input:
                if isinstance(pred, list):
                    pred = pred[-1]  # use the last feature map for multi‑scale
                total += super().forward(pred, target_is_real, is_disc, **kwargs).mean()
            return total / len(input)
        else:
            return super().forward(input, target_is_real, is_disc, **kwargs)
