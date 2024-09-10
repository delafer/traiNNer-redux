# https://github.com/umzi2/MoSR/blob/master/neosr/archs/mosr_arch.py
import torch
from timm.layers import DropPath
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from traiNNer.archs.arch_util import DySample
from traiNNer.utils.registry import ARCH_REGISTRY


class GPS(nn.Module):
    """Geo ensemble PixelShuffle"""

    def __init__(
        self,
        dim: int,
        scale: int,
        out_ch: int = 3,
        # Own parameters
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_to_k = nn.Conv2d(
            dim, scale * scale * out_ch * 8, kernel_size, 1, kernel_size // 2
        )
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x: Tensor) -> Tensor:
        rgb = self._geo_ensemble(x)
        rgb = self.ps(rgb)
        return rgb

    def _geo_ensemble(self, x: Tensor) -> Tensor:
        x = self.in_to_k(x)
        x = x.reshape(x.shape[0], 8, -1, x.shape[-2], x.shape[-1])
        x = x.mean(dim=1)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvBlock(nn.Module):
    r"""https://github.com/joshyZhou/AST/blob/main/model.py#L22"""

    def __init__(self, in_channel: int, out_channel: int, strides: int = 1) -> None:
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.Mish(),
            nn.Conv2d(
                out_channel, out_channel, kernel_size=3, stride=strides, padding=1
            ),
            nn.Mish(),
        )
        self.conv11 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=strides, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0,
        kernel_size: int = 7,
        drop_path: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size,
            1,
            kernel_size // 2,
            groups=conv_channels,
        )
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0 or not self.training
            else nn.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        x = self.drop_path(x)
        return x + (shortcut - 0.5)


@ARCH_REGISTRY.register()
class MoSR(nn.Module):
    """Mamba Out Super-Resolution"""

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        scale: int = 4,
        n_block: int = 24,
        dim: int = 64,
        upsampler: str = "pixelshuffle",  # "pixelshuffle" "dysample" "geoensemblepixelshuffle"
        drop_path: float = 0.0,
        kernel_size: int = 7,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        if upsampler in ["pixelshuffle", "geoensemblepixelshuffle"]:
            out_ch = in_ch
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, n_block)]
        self.gblocks = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [
                GatedCNNBlock(
                    dim=dim,
                    expansion_ratio=expansion_ratio,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dp_rates[index],
                )
                for index in range(n_block)
            ]
            + [
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.Mish(),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Mish(),
                nn.Conv2d(dim, dim, 1, 1),
            ]
        )

        self.shortcut = ConvBlock(in_ch, dim)

        if upsampler == "pixelshuffle":
            self.upsampler = nn.Sequential(
                nn.Conv2d(dim, out_ch * (scale**2), 3, 1, 1), nn.PixelShuffle(scale)
            )
        elif upsampler == "geoensemblepixelshuffle":
            self.upsampler = GPS(dim, scale, out_ch)
        elif upsampler == "dysample":
            self.upsampler = DySample(dim, out_ch, scale)
        else:
            raise ValueError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                "pixelshuffle" "dysample" "geoensemblepixelshuffle"'
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.gblocks(x) + (self.shortcut(x) - 0.5)
        return self.upsampler(x)


@ARCH_REGISTRY.register()
def mosr_t(**kwargs) -> MoSR:
    return MoSR(n_block=5, dim=48, expansion_ratio=1.5, conv_ratio=1.00, **kwargs)
