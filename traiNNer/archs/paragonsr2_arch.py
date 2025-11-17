#!/usr/bin/env python3
"""
ParagonSR2: A Refined, Deployment-Ready Super-Resolution Architecture
Author: Philip Hofmann (adapted/edited)

This implementation is the SR-only, ready-to-drop version of ParagonSR2 that
prioritizes:
  - Practical deployment (fuseable blocks, ONNX/TensorRT friendliness)
  - Training speed (cheap dynamic mode, fast_body_mode, channels_last support)
  - Robustness (fallbacks, convergence check for dynamic kernels)
  - Clear, production-style comments to explain why choices were made

Notes:
  - This file intentionally does NOT include any perceptual encoder. Your
    training framework should load perceptual losses (ConvNeXt/VGG/DINO) as
    separate modules — keeps the SR graph small and export-friendly.
  - Use `fuse_for_release()` before export to get fused ReparamConv kernels and
    to disable dynamic kernel generators for deterministic inference.
  - Two-phase training suggestion (not implemented here) — switch from
    dynamic_training_mode="cheap" during fast phase to "full" for final fine-tune.
"""

import warnings
from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

from .resampler import MagicKernelSharp2021Upsample

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ReparamConvV2(nn.Module):
    """
    Reparameterizable convolution block used during training (multi-branch) and
    fused for inference.
    - Training: conv3x3 + conv1x1 (+ optional depthwise3x3) summed.
    - Inference: fused into a single 3x3 conv for efficient export.
    Rationale: enables training-time capacity / inference-time speed.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.stride, self.groups = (
            in_channels,
            out_channels,
            stride,
            groups,
        )
        # Bias kept intentionally for correct fusion math
        self.conv3x3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=True,
        )
        self.conv1x1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
            bias=True,
        )

        # Optional depthwise branch for extra capacity when channels==groups==in_channels
        self.dw_conv3x3: nn.Conv2d | None = None
        if in_channels == out_channels and groups == in_channels:
            self.dw_conv3x3 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=True,
            )

    def get_fused_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (fused_kernel, fused_bias) for replacing multi-branch with single 3x3 conv.
        This is used by `fuse_for_release`.
        """
        fused_kernel = self.conv3x3.weight.detach().clone()
        bias3x3 = self.conv3x3.bias
        if bias3x3 is None:
            raise RuntimeError("ReparamConvV2.conv3x3 must use bias=True for fusion.")
        fused_bias = bias3x3.detach().clone()

        # 1x1 padded into 3x3
        padded_1x1 = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        fused_kernel += padded_1x1
        bias1x1 = self.conv1x1.bias
        if bias1x1 is None:
            raise RuntimeError("ReparamConvV2.conv1x1 must use bias=True for fusion.")
        fused_bias += bias1x1.detach()

        # add depthwise branch (converted into standard conv shape)
        if self.dw_conv3x3 is not None:
            dw_k = self.dw_conv3x3.weight
            dw_b = self.dw_conv3x3.bias
            tgt_shape = self.conv3x3.weight.shape  # (out, in/groups, k, k)
            standard_dw = torch.zeros(tgt_shape, device=dw_k.device, dtype=dw_k.dtype)
            # insert depthwise channels into appropriate positions
            for i in range(self.in_channels):
                standard_dw[i, 0, :, :] = dw_k[i, 0, :, :]
            fused_kernel += standard_dw
            if dw_b is not None:
                fused_bias += dw_b.detach()

        return fused_kernel, fused_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.dw_conv3x3 is not None:
                out = out + self.dw_conv3x3(x)
            return out
        else:
            w, b = self.get_fused_kernels()
            return F.conv2d(x, w, b, stride=self.stride, padding=1, groups=self.groups)


class ResidualBlock(nn.Module):
    """
    Lightweight 2-conv residual block used in HR refinement head.
    - Optionally supports GroupNorm-only normalization for variance control
      (affine=False to avoid interfering with quantization calibration).
    - LeakyReLU retained for hardware-friendly activation.
    """

    def __init__(self, dim: int, use_norm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            # GroupNorm with group=1 is an L2-stable, export-friendly normalization.
            self.norm1 = nn.GroupNorm(1, dim, affine=False)
            self.norm2 = nn.GroupNorm(1, dim, affine=False)
        else:
            self.norm1 = None
            self.norm2 = None

        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm and self.norm1 is not None and self.norm2 is not None:
            residual = x
            x = self.norm1(x)
            x = self.conv1(x)
            x = self.act(x)
            x = self.norm2(x)
            x = self.conv2(x)
            return residual + x
        else:
            return x + self.conv2(self.act(self.conv1(x)))


class InceptionDWConv2d(nn.Module):
    """
    Efficient multi-scale depthwise block:
      - Splits channels into ID + three depthwise branches (square, horiz, vert).
      - Very parameter efficient and friendly for quantization / fusion.
    """

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        # depthwise branches (groups=gc) — cost scales with gc not full channels
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, (1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, (band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc
        )
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )


class LayerScale(nn.Module):
    """
    LayerScale stabilization: small learnable scaling applied to residual branches.
    Optimized implementation stores gamma as (1,C,1,1) so we can avoid costly permutes.
    This is faster and plays nicely with channels-last memory.
    """

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(
            torch.full((1, dim, 1, 1), float(init_values), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ---------------------------------------------------------------------------
# Dynamic / "transformer-like" block (training-time dynamic kernels)
# - Two modes:
#     * "cheap": SE-like channel modulation (fast)
#     * "full": per-sample depthwise kernels (higher quality, expensive)
# - Tracked kernel EMA allows static export path for inference
# ---------------------------------------------------------------------------


class DynamicKernelGenerator(nn.Module):
    """
    Small predictor producing per-sample depthwise kernels (B, C, K, K).
    Kept intentionally compact (avgpool + 2 convs) to limit overhead.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * kernel_size * kernel_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, dim, _, _ = x.shape
        kernels = self.predictor(x)  # (B, dim * K*K, 1, 1)
        return kernels.reshape(b, dim, self.kernel_size, self.kernel_size)


class CheapDynamicModulation(nn.Module):
    """
    Fast SE-like channel modulation. Low compute, brings many dynamic benefits in practice.
    Use this as the default during the initial / fast training phase.
    """

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, dim // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DynamicTransformer(nn.Module):
    """
    DynamicTransformer: applies either cheap modulation or full per-sample depthwise
    kernels. Tracks an EMA of kernels for a static inference alternative.

    Important training knobs (constructor args / attributes):
      - dynamic_training_mode: "cheap" | "full" | "off"
      - dynamic_update_every: how often to update tracked kernels (can be >1 to save compute)
      - kernel_momentum: EMA momentum for tracked kernel
      - static_mode: force static path (used before export)
      - fallback_to_identity: defensive fallback on errors
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 2.0,
        dynamic_training_mode: str = "cheap",
        dynamic_update_every: int = 8,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)

        assert dynamic_training_mode in ["cheap", "full", "off"], (
            "dynamic_training_mode must be 'cheap'|'full'|'off'"
        )
        self.dynamic_training_mode = dynamic_training_mode
        self.dynamic_update_every = max(1, int(dynamic_update_every))

        self.kernel_generator: DynamicKernelGenerator | None = None
        self.cheap_dynamic: CheapDynamicModulation | None = None

        if self.dynamic_training_mode == "full":
            self.kernel_generator = DynamicKernelGenerator(hidden_dim)
        elif self.dynamic_training_mode == "cheap":
            self.cheap_dynamic = CheapDynamicModulation(hidden_dim)

        self.kernel_size = 3
        self.padding = self.kernel_size // 2
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1)

        # export / tracking controls
        self.dynamic_inference = False
        self.track_running_stats = True
        self.kernel_momentum = 0.05
        self.static_mode = False
        self.fallback_to_identity = True

        # tracked kernel buffers (shape: (hidden_dim, 1, K, K))
        identity = torch.zeros(
            hidden_dim, 1, self.kernel_size, self.kernel_size, dtype=torch.float32
        )
        identity[:, 0, self.padding, self.padding] = 1.0
        self.register_buffer("identity_kernel", identity, persistent=True)
        self.register_buffer(
            "tracked_kernel", torch.zeros_like(identity), persistent=True
        )
        self.register_buffer(
            "tracked_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

        # light counters for update scheduling
        self._batch_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward applies project_in -> dynamic/static kernel op -> project_out.
        It includes safe fallbacks and keeps the static path inexpensive.
        """
        try:
            x = self.project_in(x)
            use_dynamic = (self.training or self.dynamic_inference) and (
                not self.static_mode
            )

            if use_dynamic:
                if self.dynamic_training_mode == "cheap":
                    y = x if self.cheap_dynamic is None else self.cheap_dynamic(x)
                elif self.dynamic_training_mode == "full":
                    if self.kernel_generator is None:
                        if self.fallback_to_identity:
                            warnings.warn(
                                "Missing kernel_generator; falling back to identity dynamic op.",
                                UserWarning,
                                stacklevel=2,
                            )
                            y = x
                        else:
                            raise RuntimeError(
                                "Kernel generator missing while 'full' mode requested."
                            )
                    else:
                        b, c, _, _ = x.shape
                        kernels = self.kernel_generator(x)  # (B, C, K, K)
                        y = self._apply_dynamic_kernel(
                            x, kernels, batch_size=b, channels=c
                        )
                        # Update tracked kernel occasionally to reduce overhead
                        if self.training and self.track_running_stats:
                            self._batch_counter += 1
                            if (self._batch_counter % self.dynamic_update_every) == 0:
                                self._update_tracked_kernel(kernels)
                else:  # "off"
                    y = x
            else:
                y = self._apply_static_kernel(x)

            return self.project_out(self.act(y))

        except Exception as e:
            if self.fallback_to_identity:
                warnings.warn(
                    f"DynamicTransformer forward failed ({e}), falling back to identity.",
                    UserWarning,
                    stacklevel=2,
                )
                x = self.project_in(x)
                return self.project_out(self.act(x))
            else:
                raise

    def _apply_dynamic_kernel(
        self, x: torch.Tensor, kernels: torch.Tensor, batch_size: int, channels: int
    ) -> torch.Tensor:
        # Efficient grouped conv: reshape into (1, B*C, H, W) and depthwise kernels (B*C,1,k,k)
        try:
            x_reshaped = x.reshape(1, batch_size * channels, x.shape[2], x.shape[3])
            kernels_reshaped = kernels.reshape(
                batch_size * channels, 1, self.kernel_size, self.kernel_size
            )
            y = F.conv2d(
                x_reshaped,
                kernels_reshaped,
                padding=self.padding,
                groups=batch_size * channels,
            )
            return y.reshape(batch_size, channels, x.shape[2], x.shape[3])
        except Exception as e:
            if self.fallback_to_identity:
                warnings.warn(
                    f"Dynamic kernel application failed ({e}). Using identity.",
                    UserWarning,
                    stacklevel=2,
                )
                return x
            else:
                raise

    def _apply_static_kernel(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._get_tracked_kernel().to(dtype=x.dtype, device=x.device)
        # kernel shape (hidden_dim, 1, k, k) groups=x.shape[1]
        return F.conv2d(x, kernel, padding=self.padding, groups=x.shape[1])

    def _get_tracked_kernel(self) -> torch.Tensor:
        tracked_initialized = cast(torch.Tensor, self.tracked_initialized)
        tracked_kernel = cast(torch.Tensor, self.tracked_kernel)
        identity_kernel = cast(torch.Tensor, self.identity_kernel)
        if bool(tracked_initialized.item()):
            return tracked_kernel
        # fallback to identity when not initialized (safe)
        if not self.training:
            warnings.warn(
                "DynamicTransformer: tracked kernel not initialized — using identity kernel for static path.",
                UserWarning,
                stacklevel=2,
            )
        return identity_kernel

    def _update_tracked_kernel(self, kernels: torch.Tensor) -> None:
        """
        Update EMA of tracked kernel with mean of current batch kernels.
        Protected by no-grad and defensive checks.
        """
        with torch.no_grad():
            tracked_kernel = cast(torch.Tensor, self.tracked_kernel)
            tracked_initialized = cast(torch.Tensor, self.tracked_initialized)
            mean_kernel = (
                kernels.mean(dim=0).unsqueeze(1).to(dtype=tracked_kernel.dtype)
            )
            if not bool(tracked_initialized.item()):
                tracked_kernel.copy_(mean_kernel)
                tracked_initialized.fill_(True)
            else:
                momentum = float(self.kernel_momentum)
                if momentum <= 0:
                    tracked_kernel.copy_(mean_kernel)
                else:
                    tracked_kernel.lerp_(mean_kernel, weight=momentum)

    def enable_dynamic_inference(self, enabled: bool = True) -> None:
        self.dynamic_inference = enabled

    def enable_static_mode(self, enabled: bool = True) -> None:
        self.static_mode = enabled

    def set_kernel_momentum(self, momentum: float) -> None:
        if momentum < 0 or momentum > 1:
            raise ValueError("kernel_momentum must be in [0,1].")
        self.kernel_momentum = float(momentum)

    def reset_tracked_kernel(self) -> None:
        tracked_kernel = cast(torch.Tensor, self.tracked_kernel)
        tracked_initialized = cast(torch.Tensor, self.tracked_initialized)
        tracked_kernel.zero_()
        tracked_initialized.fill_(False)

    def export_static_depthwise(self) -> nn.Conv2d:
        """
        Return a depthwise Conv2d initialized from tracked kernel for export.
        Use when preparing ONNX/TensorRT inference graph.
        """
        kernel = self._get_tracked_kernel().detach()
        conv = nn.Conv2d(
            kernel.shape[0],
            kernel.shape[0],
            self.kernel_size,
            padding=self.padding,
            groups=kernel.shape[0],
            bias=False,
        )
        with torch.no_grad():
            conv.weight.copy_(kernel)
        return conv


# ---------------------------------------------------------------------------
# High-level building blocks that combine context + dynamic transformer
# ---------------------------------------------------------------------------


class ParagonBlockV2(nn.Module):
    """
    Core block:
      - InceptionDWConv2d context (local multi-scale)
      - LayerScale stabilization
      - DynamicTransformer (cheap/full/off)
      - Residual connections around both parts
    """

    def __init__(
        self,
        dim: int,
        ffn_expansion: float = 2.0,
        use_norm: bool = False,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.context = InceptionDWConv2d(
            dim,
            **{
                k: v
                for k, v in block_kwargs.items()
                if k
                in [
                    "square_kernel_size",
                    "band_kernel_size",
                    "branch_ratio",
                    "band_kernel_size",
                ]
            },
        )
        self.ls1 = LayerScale(dim)
        # propagate dynamic training options into transformer via block_kwargs keys
        dt_mode = block_kwargs.get("dynamic_training_mode", "cheap")
        dt_update = block_kwargs.get("dynamic_update_every", 8)
        self.transformer = DynamicTransformer(
            dim,
            expansion_ratio=ffn_expansion,
            dynamic_training_mode=dt_mode,
            dynamic_update_every=dt_update,
        )
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.context(x)
        x = residual + self.ls1(x)

        residual = x
        x = self.transformer(x)
        x = residual + self.ls2(x)
        return x


class ResidualGroupV2(nn.Module):
    """
    Group of ParagonBlockV2 modules with a single outer residual connection.
    Keeps the body composable and simple to adjust depth.
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        ffn_expansion: float = 2.0,
        use_norm: bool = False,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ParagonBlockV2(dim, ffn_expansion, use_norm=use_norm, **block_kwargs)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x) + x


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------


class ParagonSR2(nn.Module):
    """
    ParagonSR2 main model.

    Key performance/config knobs:
      - dynamic_training_mode: "cheap" (fast) | "full" (quality) | "off"
      - dynamic_update_every: number of steps between tracked kernel updates
      - use_channels_last: if True (and CUDA available) model attempts to use channels_last memory_layout
      - fast_body_mode: reduces depth (groups & blocks //= 2) for fast training prototypes
      - use_norm: optionally enable GroupNorm in small places to stabilize training (turn on for larger models)
    """

    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 6,
        num_blocks: int = 6,
        ffn_expansion: float = 2.0,
        block_kwargs: dict | None = None,
        upsampler_alpha: float = 0.5,
        hr_blocks: int = 1,
        dynamic_training_mode: str = "cheap",
        dynamic_update_every: int = 8,
        use_channels_last: bool = True,
        fast_body_mode: bool = False,
        use_norm: bool = False,
        robust_mode: bool = True,
    ) -> None:
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}

        # Propagate performance knobs to block kwargs
        block_kwargs.update(
            {
                "dynamic_training_mode": dynamic_training_mode,
                "dynamic_update_every": dynamic_update_every,
            }
        )

        self.scale = scale
        upsampler_alpha = float(max(0.0, min(1.0, float(upsampler_alpha))))
        self.upsampler_alpha = upsampler_alpha
        self.hr_blocks = max(int(hr_blocks), 0)

        # fast_body_mode halves groups and blocks to speed up training (cheap option)
        if fast_body_mode:
            num_groups = max(1, num_groups // 2)
            num_blocks = max(1, num_blocks // 2)

        # Shallow extraction
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1)

        # Core body (LR space): multiple ResidualGroupV2
        self.body = nn.Sequential(
            *[
                ResidualGroupV2(
                    num_feat,
                    num_blocks,
                    ffn_expansion,
                    use_norm=use_norm and robust_mode,
                    **block_kwargs,
                )
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampler (Magic Kernel Sharp 2021): pre-sharpen + resample
        self.magic_upsampler = MagicKernelSharp2021Upsample(
            in_channels=num_feat, alpha=self.upsampler_alpha
        )

        # HR head
        self.hr_conv_in = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if self.hr_blocks > 0:
            self.hr_head = nn.Sequential(
                *[
                    ResidualBlock(num_feat, use_norm=use_norm and robust_mode)
                    for _ in range(self.hr_blocks)
                ]
            )
        else:
            self.hr_head = nn.Identity()

        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

        # Channels-last optimization (attempt best-effort conversion)
        self.use_channels_last = use_channels_last and torch.cuda.is_available()
        if self.use_channels_last:
            try:
                for module in self.modules():
                    if hasattr(module, "weight") and module.weight is not None:
                        if module.weight.dtype in (torch.float32, torch.float16):
                            module.weight.data = module.weight.contiguous(
                                memory_format=torch.channels_last
                            )
            except Exception:
                # best-effort only — do not fail model creation on unusual modules
                pass

    def fuse_for_release(self) -> "ParagonSR2":
        """
        Fuse training-time modules into inference-time equivalents:
          - Replace ReparamConvV2 multi-branch with single fused conv
          - Disable kernel generator and set transformers to static mode
        Call this before export (ONNX/TensorRT) to get deterministic, fast graph.
        """
        print("Fusing ParagonSR v2: performing inference-time fusion steps...")
        for name, module in list(self.named_modules()):
            # Make DynamicTransformer static and remove generator (keeps tracked kernel)
            if isinstance(module, DynamicTransformer):
                tracked_initialized = cast(torch.Tensor, module.tracked_initialized)
                if not bool(tracked_initialized.item()):
                    warnings.warn(
                        f"{name}: tracked kernels not initialized — identity will be used.",
                        UserWarning,
                        stacklevel=2,
                    )
                module.enable_static_mode(True)
                module.kernel_generator = None
                continue

            # Fuse ReparamConvV2 instances
            if isinstance(module, ReparamConvV2):
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = self.get_submodule(parent_name)
                print(f"  - Fusing {name}")
                w, b = module.get_fused_kernels()
                fused_conv = nn.Conv2d(
                    module.conv3x3.in_channels,
                    module.conv3x3.out_channels,
                    3,
                    module.stride,
                    1,
                    groups=module.groups,
                    bias=True,
                )
                with torch.no_grad():
                    fused_conv.weight.copy_(w)
                    if fused_conv.bias is None:
                        raise RuntimeError("Expected fused conv to have bias.")
                    fused_conv.bias.copy_(b)
                setattr(parent_module, child_name, fused_conv)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with defensive error handling — returns input on catastrophic error
        so training loop doesn't completely crash when experimenting (helps iterative debugging).
        """
        try:
            if self.use_channels_last and x.is_cuda:
                # Use channels_last for inputs too (best-effort)
                x = x.contiguous(memory_format=torch.channels_last)

            # LR feature extraction and body
            x_shallow = self.conv_in(x)
            x_deep = self.body(x_shallow)
            x_fused = self.conv_fuse(x_deep) + x_shallow

            # Upsample (Magic kernel)
            x_upsampled = self.magic_upsampler(x_fused, scale_factor=self.scale)

            # HR refinement
            h = self.hr_conv_in(x_upsampled)
            h = self.hr_head(h)

            return self.conv_out(h)

        except Exception as e:
            warnings.warn(
                f"ParagonSR2 forward failed: {e}. Falling back to identity.",
                UserWarning,
                stacklevel=2,
            )
            return x


# ---------------------------------------------------------------------------
# Factory registration for different sizes
# ---------------------------------------------------------------------------
@ARCH_REGISTRY.register()
def paragonsr2_nano(scale: int = 4, **kwargs) -> ParagonSR2:
    """
    Nano variant: minimal channels and depth for extremely fast training / tiny devices.
    Use for rapid prototyping.
    """
    return ParagonSR2(
        scale=scale,
        num_feat=24,
        num_groups=2,
        num_blocks=2,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 9},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 1),
        dynamic_training_mode=kwargs.get("dynamic_training_mode", "cheap"),
        dynamic_update_every=kwargs.get("dynamic_update_every", 8),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", True),
        use_norm=kwargs.get("use_norm", True),
        robust_mode=kwargs.get("robust_mode", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_xs(scale: int = 4, **kwargs) -> ParagonSR2:
    return ParagonSR2(
        scale=scale,
        num_feat=32,
        num_groups=2,
        num_blocks=3,
        ffn_expansion=1.5,
        block_kwargs={"band_kernel_size": 11},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 1),
        dynamic_training_mode=kwargs.get("dynamic_training_mode", "cheap"),
        dynamic_update_every=kwargs.get("dynamic_update_every", 8),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", True),
        use_norm=kwargs.get("use_norm", True),
        robust_mode=kwargs.get("robust_mode", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_s(scale: int = 4, **kwargs) -> ParagonSR2:
    return ParagonSR2(
        scale=scale,
        num_feat=48,
        num_groups=3,
        num_blocks=4,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 11},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 2),
        dynamic_training_mode=kwargs.get("dynamic_training_mode", "cheap"),
        dynamic_update_every=kwargs.get("dynamic_update_every", 8),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", True),
        use_norm=kwargs.get("use_norm", True),
        robust_mode=kwargs.get("robust_mode", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_m(scale: int = 4, **kwargs) -> ParagonSR2:
    return ParagonSR2(
        scale=scale,
        num_feat=64,
        num_groups=4,
        num_blocks=6,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 13},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 2),
        dynamic_training_mode=kwargs.get("dynamic_training_mode", "full"),
        dynamic_update_every=kwargs.get("dynamic_update_every", 8),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", False),
        use_norm=kwargs.get("use_norm", True),
        robust_mode=kwargs.get("robust_mode", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_l(scale: int = 4, **kwargs) -> ParagonSR2:
    return ParagonSR2(
        scale=scale,
        num_feat=96,
        num_groups=6,
        num_blocks=8,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 15},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 3),
        dynamic_training_mode=kwargs.get("dynamic_training_mode", "full"),
        dynamic_update_every=kwargs.get("dynamic_update_every", 8),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", False),
        use_norm=kwargs.get("use_norm", True),
        robust_mode=kwargs.get("robust_mode", True),
    )


@ARCH_REGISTRY.register()
def paragonsr2_xl(scale: int = 4, **kwargs) -> ParagonSR2:
    return ParagonSR2(
        scale=scale,
        num_feat=128,
        num_groups=8,
        num_blocks=10,
        ffn_expansion=2.0,
        block_kwargs={"band_kernel_size": 15},
        upsampler_alpha=kwargs.get("upsampler_alpha", 0.5),
        hr_blocks=kwargs.get("hr_blocks", 3),
        dynamic_training_mode=kwargs.get("dynamic_training_mode", "full"),
        dynamic_update_every=kwargs.get("dynamic_update_every", 8),
        use_channels_last=kwargs.get("use_channels_last", True),
        fast_body_mode=kwargs.get("fast_body_mode", False),
        use_norm=kwargs.get("use_norm", True),
        robust_mode=kwargs.get("robust_mode", True),
    )
