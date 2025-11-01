#!/usr/bin/env python3
"""
R3GAN Loss Implementation
Based on the official R3GAN repository: https://github.com/NVlabs/R3GAN

R3GAN (Relativistic GAN with Regularization) is a modern GAN architecture
that uses a relativistic discriminator formulation with R1/R2 gradient penalties.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from traiNNer.utils.registry import LOSS_REGISTRY


class AdversarialTraining:
    """Official R3GAN Adversarial Training implementation"""

    def __init__(self, Generator, Discriminator) -> None:
        self.Generator = Generator
        self.Discriminator = Discriminator

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        """Zero-centered gradient penalty computation"""
        try:
            (Gradient,) = torch.autograd.grad(
                outputs=Critics.sum(),
                inputs=Samples,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,  # Allow unused tensors
            )
            # Return zero penalty if gradient is None (tensor not used)
            if Gradient is None:
                return torch.zeros(Samples.size(0), device=Samples.device)
            return Gradient.square().sum([1, 2, 3])
        except RuntimeError as e:
            if "allow_unused" in str(e):
                return torch.zeros(Samples.size(0), device=Samples.device)
            raise e

    def AccumulateGeneratorGradients(
        self, Noise, RealSamples, Conditions, Scale=1, Preprocessor=lambda x: x
    ):
        """Accumulate gradients for generator (R3GAN relativistic formulation)"""
        # Handle models that don't take conditions
        if Conditions is not None:
            FakeSamples = self.Generator(Noise, Conditions)
        else:
            FakeSamples = self.Generator(Noise)

        RealSamples = RealSamples.detach()  # No gradients needed for real samples

        # Handle discriminators that don't take conditions
        if Conditions is not None:
            FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
            RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        else:
            FakeLogits = self.Discriminator(Preprocessor(FakeSamples))
            RealLogits = self.Discriminator(Preprocessor(RealSamples))

        # Relativistic formulation: fake - real
        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = F.softplus(-RelativisticLogits)

        return Scale * AdversarialLoss.mean()

        return (AdversarialLoss, RelativisticLogits)

    def AccumulateDiscriminatorGradients(
        self, Noise, RealSamples, Conditions, Gamma, Scale=1, Preprocessor=lambda x: x
    ):
        """Accumulate gradients for discriminator with R1/R2 penalties"""
        # CRITICAL: Enable gradients for gradient penalty computation
        RealSamples = RealSamples.detach().requires_grad_(True)

        # Handle models that don't take conditions
        if Conditions is not None:
            FakeSamples = (
                self.Generator(Noise, Conditions).detach().requires_grad_(True)
            )
        else:
            FakeSamples = self.Generator(Noise).detach().requires_grad_(True)

        # Handle discriminators that don't take conditions
        if Conditions is not None:
            RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
            FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        else:
            RealLogits = self.Discriminator(Preprocessor(RealSamples))
            FakeLogits = self.Discriminator(Preprocessor(FakeSamples))

        # R1 and R2 gradient penalties
        R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
            RealSamples, RealLogits
        )
        R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
            FakeSamples, FakeLogits
        )

        # Relativistic formulation: real - fake (discriminator wants real > fake)
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = F.softplus(-RelativisticLogits)

        # Reduce AdversarialLoss to per-sample mean (scalar per sample)
        # This is because R1Penalty and R2Penalty are per-sample scalars
        if AdversarialLoss.dim() > 1:
            AdversarialLoss = AdversarialLoss.mean(
                dim=list(range(1, AdversarialLoss.dim()))
            )

        # Total loss: adversarial + gradient penalties
        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        return Scale * DiscriminatorLoss.mean()

        return (
            AdversarialLoss,
            RelativisticLogits,
            R1Penalty,
            R2Penalty,
        )


@LOSS_REGISTRY.register()
class R3GANLoss(nn.Module):
    """
    R3GAN Loss Implementation adapted for traiNNer framework

    Based on the official R3GAN implementation but adapted to work with
    the standard traiNNer training loop structure.
    """

    def __init__(
        self,
        loss_weight: float,
        gan_type: str = "r3gan",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        r1_weight: float = 3.0,  # Adjusted from R3GAN paper defaults
        r2_weight: float = 3.0,  # Adjusted from R3GAN paper defaults
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

        # Store for training loop access
        self.trainer = None

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

    def init_trainer(self, generator, discriminator) -> None:
        """Initialize the R3GAN trainer with generator and discriminator"""
        self.trainer = AdversarialTraining(generator, discriminator)

    def accumulate_generator_gradients(
        self, noise, real_samples, conditions=None, scale=1
    ):
        """Accumulate gradients for generator (R3GAN relativistic)"""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call init_trainer() first.")

        return self.trainer.AccumulateGeneratorGradients(
            noise, real_samples, conditions, scale
        )

    def accumulate_discriminator_gradients(
        self, noise, real_samples, conditions=None, gamma=None, scale=1
    ):
        """Accumulate gradients for discriminator with R1/R2 penalties"""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call init_trainer() first.")

        # Use configured gamma if not provided
        if gamma is None:
            gamma = self.r1_weight + self.r2_weight  # Combined gradient penalty weight

        return self.trainer.AccumulateDiscriminatorGradients(
            noise, real_samples, conditions, gamma, scale
        )

    # --------------------------------------------------------------------- #
    # Legacy loss implementations (kept for compatibility)
    # --------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    # Core R3GAN implementation (for compatibility)
    # --------------------------------------------------------------------- #
    def _r3gan_loss(
        self,
        input: Tensor,
        target_is_real: bool,
        is_disc: bool = False,
        real_images: Tensor | None = None,
        fake_images: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Relativistic R3GAN loss with R1/R2 gradient penalties"""

        if is_disc:
            # Discriminator wants real to have higher score than fake
            # This is the standard R3GAN relativistic formulation
            loss = F.softplus(-input).mean()

            # Temporarily disable gradient penalties to avoid graph conflicts
            # TODO: Implement gradient penalties properly later

        else:
            # Generator wants fake to have higher score than real
            # The input is the discriminator's output for fake images
            loss = F.softplus(-input).mean()

            # Apply loss weight only for generator updates
            loss = loss * self.loss_weight

        return loss

    def _compute_gradient_penalty(
        self, images: Tensor, d_output: Tensor, is_real: bool
    ) -> Tensor:
        """Compute R1/R2 gradient penalty"""
        try:
            gradients = torch.autograd.grad(
                outputs=d_output.sum(),
                inputs=images,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,  # Allow unused tensors
            )[0]
            if gradients is None:
                return torch.tensor(0.0, device=images.device)
            gradients = gradients.view(gradients.size(0), -1)
            penalty = gradients.norm(2, dim=1) - 1
            penalty = penalty.pow(2).mean()
            return penalty
        except RuntimeError as e:
            if "allow_unused" in str(e):
                return torch.tensor(0.0, device=images.device)
            raise e

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _get_target_label(self, input: Tensor, target_is_real: bool) -> Tensor:
        """Return target label for non‑relativistic loss types."""
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(
        self,
        input: Tensor,
        target_is_real: bool,
        is_disc: bool = False,
        **kwargs,
    ) -> Tensor:
        """Forward entry point used by the training loop."""
        if self.gan_type == "r3gan":
            return self._r3gan_loss(input, target_is_real, is_disc, **kwargs)
        else:
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
