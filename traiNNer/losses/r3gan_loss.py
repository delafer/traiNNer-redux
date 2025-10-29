import torch
from torch import Tensor, autograd, nn
from torch.nn import functional as F

from traiNNer.utils.registry import LOSS_REGISTRY


def r1_r2_penalty(logits: Tensor, images: Tensor, penalty_type: str = "r1") -> Tensor:
    """R1/R2 gradient penalty for relativistic GANs.

    Args:
        logits: Discriminator output logits (any shape, typically [B, ...]).
        images: Input images (real or fake) corresponding to ``logits``.
        penalty_type: ``"r1"`` for real‑image penalty, ``"r2"`` for fake‑image penalty.

    Returns:
        Scalar tensor containing the gradient penalty.
    """
    gradients = autograd.grad(
        outputs=logits.sum(),
        inputs=images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients to compute L2 norm per sample
    gradients = gradients.view(gradients.size(0), -1)
    grad_penalty = gradients.pow(2).sum(dim=1).mean()
    return grad_penalty


@LOSS_REGISTRY.register()
class R3GANLoss(nn.Module):
    """R3GAN (Relativistic GAN with Regularization) loss.

    This loss implements the relativistic discriminator formulation
    from the R3GAN paper (NeurIPS 2024). It supports the standard
    GAN types for backward compatibility.

    Args:
        loss_weight (float): Weight applied to the generator loss.
        gan_type (str): ``"r3gan"`` (default) or any of the legacy types
            (``"vanilla"``, ``"lsgan"``, ``"wgan"``, ``"wgan_softplus"``,
            ``"hinge"``) for compatibility.
        real_label_val (float): Unused for ``r3gan`` but kept for API compatibility.
        fake_label_val (float): Unused for ``r3gan`` but kept for API compatibility.
        r1_weight (float): Weight for the R1 gradient penalty (default: 10.0).
        r2_weight (float): Weight for the R2 gradient penalty (default: 10.0).
        use_relu (bool): If ``True`` and ``gan_type`` is ``"hinge"``, uses ``nn.ReLU``.
    """

    def __init__(
        self,
        loss_weight: float,
        gan_type: str = "r3gan",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        r1_weight: float = 10.0,
        r2_weight: float = 10.0,
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
            self.loss = nn.ReLU() if self.use_relu else nn.ReLU()
            self.loss_func = self._hinge_loss
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

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
        self, input: Tensor, target_is_real: bool, **_: dict
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
    # Core R3GAN implementation
    # --------------------------------------------------------------------- #
    def _r3gan_loss(
        self,
        input: Tensor,
        target_is_real: bool,
        is_disc: bool = False,
        real_images: Tensor | None = None,
        fake_images: Tensor | None = None,
    ) -> Tensor:
        """Relativistic loss with optional gradient penalties.

        Args:
            input: Discriminator logits for either real or fake batch
                (depending on ``target_is_real``).
            target_is_real: ``True`` if ``input`` corresponds to real images.
            is_disc: ``True`` when computing the discriminator loss.
            real_images: Tensor of real images (required for R1 penalty).
            fake_images: Tensor of fake images (required for R2 penalty).

        Returns:
            Loss tensor (already multiplied by ``loss_weight`` for the generator).
        """
        if is_disc:
            # Discriminator wants real > fake
            loss = F.softplus(-input).mean()

            # R1 penalty on real images
            if real_images is not None and self.r1_weight > 0:
                r1 = r1_r2_penalty(input, real_images, "r1")
                loss = loss + self.r1_weight * r1

            # R2 penalty on fake images
            if fake_images is not None and self.r2_weight > 0:
                r2 = r1_r2_penalty(input, fake_images, "r2")
                loss = loss + self.r2_weight * r2
        else:
            # Generator wants fake > real
            loss = F.softplus(-input).mean()

        # Apply loss weight only for generator updates
        return loss * (self.loss_weight if not is_disc else 1.0)

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
        """Forward entry point used by the training loop.

        ``kwargs`` may contain ``real_images`` and ``fake_images`` when
        ``gan_type`` is ``"r3gan"``.
        """
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


def r3gan_training_step(
    discriminator: nn.Module,
    real_images: Tensor,
    fake_images: Tensor,
    optimizer_D: torch.optim.Optimizer,
    optimizer_G: torch.optim.Optimizer,
    r1_weight: float = 10.0,
    r2_weight: float = 10.0,
    **kwargs,
) -> tuple[Tensor, Tensor, dict]:
    """Convenient helper that performs a full R3GAN training step.

    Returns:
        loss_D, loss_G, training_info
    """
    # Discriminator forward passes
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images.detach())

    # Relativistic logits
    rel_real = real_logits - fake_logits
    rel_fake = fake_logits - real_logits

    # Discriminator loss (real > fake)
    loss_D = F.softplus(-rel_real).mean()
    if r1_weight > 0:
        real_logits_pen = discriminator(real_images)
        loss_D = loss_D + r1_weight * r1_r2_penalty(real_logits_pen, real_images, "r1")
    if r2_weight > 0:
        fake_logits_pen = discriminator(fake_images.detach())
        loss_D = loss_D + r2_weight * r1_r2_penalty(fake_logits_pen, fake_images, "r2")

    # Generator loss (fake > real)
    loss_G = F.softplus(-rel_fake).mean()

    # Optimizer steps
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    training_info = {
        "loss_D": loss_D.item(),
        "loss_G": loss_G.item(),
        "real_logits": real_logits.mean().item(),
        "fake_logits": fake_logits.mean().item(),
        "rel_real": rel_real.mean().item(),
        "rel_fake": rel_fake.mean().item(),
    }

    return loss_D, loss_G, training_info
