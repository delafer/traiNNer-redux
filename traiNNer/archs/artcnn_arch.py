# https://github.com/umzi2/ArtCNN-PyTorch/blob/master/neosr/archs/artcnn_arch.py

from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


class DepthToSpace(nn.Module):
    def __init__(self, filters: int, out_ch: int, kernel_size: int, scale: int) -> None:
        super().__init__()

        self.upscale = nn.Sequential(
            nn.Conv2d(filters, out_ch * (scale**2), kernel_size, 1, kernel_size // 2),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upscale(x)


class ActConv(nn.Sequential):
    def __init__(
        self, filters: int, kernel_size: int, act: type[nn.Module] = nn.ReLU
    ) -> None:
        super().__init__(
            nn.Conv2d(filters, filters, kernel_size, 1, kernel_size // 2), act()
        )


class ResBlock(nn.Module):
    def __init__(
        self, filters: int, kernel_size: int, act: type[nn.Module] = nn.ReLU
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ActConv(filters, kernel_size, act),
            ActConv(filters, kernel_size, act),
            nn.Conv2d(filters, filters, kernel_size, 1, kernel_size // 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.conv(x)
        return x + res


@ARCH_REGISTRY.register()
class ArtCNN(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        scale: int = 4,
        filters: int = 96,
        n_block: int = 16,
        kernel_size: int = 3,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, filters, kernel_size, 1, kernel_size // 2)
        self.res_block = nn.Sequential(
            *[ResBlock(filters, kernel_size, act) for _ in range(n_block)]
            + [nn.Conv2d(filters, filters, kernel_size, 1, kernel_size // 2)]
        )
        self.depth_to_space = DepthToSpace(filters, in_ch, kernel_size, scale)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.res_block(x) + x
        return self.depth_to_space(x)


@ARCH_REGISTRY.register()
def artcnn_r16f96(scale: int = 4, **kwargs) -> ArtCNN:
    return ArtCNN(scale=scale, n_block=16, filters=96, **kwargs)


@ARCH_REGISTRY.register()
def artcnn_r8f64(scale: int = 4, **kwargs) -> ArtCNN:
    return ArtCNN(scale=scale, n_block=8, filters=64, **kwargs)
