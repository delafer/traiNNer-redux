from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.transforms import functional as TF

from traiNNer.losses.chc_loss import CHCLoss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ConsistencyLoss(nn.Module):
    """Brightness and chroma consistency loss based on Oklab chroma and CIE L*."""

    def __init__(
        self,
        loss_weight: float = 0.25,
        criterion: str = "chc",
        blur: bool = True,
        blur_kernel_size: int = 21,
        blur_sigma: float = 3.0,
        saturation: float = 1.0,
        brightness: float = 0.9,
        cosim: bool = True,
        cosim_weight: float = 0.5,
        chc_lambda: float = 0.012,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.use_blur = blur
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.saturation = float(saturation)
        self.brightness = float(brightness)
        self.use_cosim = cosim
        self.cosim_weight = float(cosim_weight)

        if criterion == "l1":
            self.criterion: nn.Module = nn.L1Loss()
        elif criterion == "huber":
            self.criterion = nn.HuberLoss()
        elif criterion == "chc":
            self.criterion = CHCLoss(
                loss_weight=1.0,
                reduction="mean",
                criterion="huber",
                loss_lambda=chc_lambda,
            )
        else:
            raise NotImplementedError(
                f"Unsupported consistency criterion '{criterion}'."
            )

        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-12)

        self.register_buffer(
            "chroma_mean",
            torch.tensor((0.5, 0.5), dtype=torch.float32).view(1, 2, 1, 1),
        )
        self.register_buffer(
            "rgb2y", torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
        )

    @staticmethod
    def _lin_rgb(img: Tensor) -> Tensor:
        return torch.where(
            img <= 0.04045,
            img / 12.92,
            ((img + 0.055) / 1.055).clamp(min=1e-8).pow(2.4),
        )

    def _rgb_to_oklab_chroma(self, img: Tensor) -> Tensor:
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"Oklab chroma expects (B,3,H,W); got {tuple(img.shape)}")

        img_lin = self._lin_rgb(img)

        r, g, b = img_lin[:, 0], img_lin[:, 1], img_lin[:, 2]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = l.sign() * l.abs().clamp(min=1e-8).pow(1 / 3)
        m_ = m.sign() * m.abs().clamp(min=1e-8).pow(1 / 3)
        s_ = s.sign() * s.abs().clamp(min=1e-8).pow(1 / 3)

        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return torch.stack([a, b_], dim=1)

    def _rgb_to_l_star(self, img: Tensor) -> Tensor:
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"L* expects (B,3,H,W); got {tuple(img.shape)}")

        img_lin = self._lin_rgb(img.permute(0, 2, 3, 1))
        y = torch.tensordot(img_lin, self.rgb2y, dims=([-1], [0]))

        y = torch.where(
            y <= (216.0 / 24389.0),
            y * (24389.0 / 27.0),
            y.clamp(min=1e-8).pow(1 / 3) * 116.0 - 16.0,
        )
        return torch.clamp(y / 100.0, 0.0, 1.0)

    def forward(self, net_output: Tensor, gt: Tensor) -> Tensor:
        net_output = torch.clamp(net_output, 1.0 / 255.0, 1.0)
        gt = torch.clamp(gt, 1.0 / 255.0, 1.0)

        if self.use_blur:
            net_blur = torch.clamp(
                TF.gaussian_blur(
                    net_output,
                    [self.blur_kernel_size, self.blur_kernel_size],
                    [self.blur_sigma, self.blur_sigma],
                ),
                0.0,
                1.0,
            )
            gt_blur = torch.clamp(
                TF.gaussian_blur(
                    gt,
                    [self.blur_kernel_size, self.blur_kernel_size],
                    [self.blur_sigma, self.blur_sigma],
                ),
                0.0,
                1.0,
            )
        else:
            net_blur = net_output
            gt_blur = gt

        input_luma = self._rgb_to_l_star(net_blur)
        target_luma = self._rgb_to_l_star(gt_blur) * self.brightness

        input_chroma = self._rgb_to_oklab_chroma(net_output)
        target_chroma = self._rgb_to_oklab_chroma(gt) * self.saturation

        input_chroma = torch.clamp(input_chroma + self.chroma_mean, 0.0, 1.0)
        target_chroma = torch.clamp(target_chroma + self.chroma_mean, 0.0, 1.0)

        data_term = self.criterion(input_luma, target_luma) + self.criterion(
            input_chroma, target_chroma
        )

        if self.use_cosim:
            luma_cosim = (
                1.0
                - self.cosine(input_luma.unsqueeze(1), target_luma.unsqueeze(1)).mean()
            )
            chroma_cosim = 1.0 - self.cosine(input_chroma, target_chroma).mean()
            cosim_term = self.cosim_weight * (luma_cosim + chroma_cosim)
            loss = data_term + cosim_term
        else:
            loss = data_term

        return loss * self.loss_weight
