import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "timm is required for ConvNeXtPerceptualLoss. "
        "Please install timm or remove convnextperceptualloss from your config."
    ) from exc


def _create_convnext_tiny_features() -> nn.Module:
    """Create a ConvNeXt-Tiny backbone with feature outputs only.

    Prefers the higher-quality fb_in22k_ft_in1k weights when available,
    falls back to the standard ImageNet-1k convnext_tiny otherwise.
    """
    # Prefer 22k-pretrained, 1k-finetuned variant for richer features if present.
    model_names: list[str] = ["convnext_tiny.fb_in22k_ft_in1k", "convnext_tiny"]

    last_err: Exception | None = None
    for name in model_names:
        try:
            model = timm.create_model(
                name,
                pretrained=True,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            return model
        except Exception as e:  # pragma: no cover - environment dependent
            last_err = e

    raise RuntimeError(
        f"Unable to create ConvNeXt-Tiny backbone for ConvNeXtPerceptualLoss. "
        f"Last error: {last_err}"
    )


@LOSS_REGISTRY.register()
class ConvNeXtPerceptualLoss(nn.Module):
    """ConvNeXt-Tiny based perceptual loss for SISR.

    Design goals:
    - Modern replacement for legacy VGG perceptual loss.
    - Use shallow/mid-level ConvNeXt-Tiny features for low/mid-level structure:
      avoids over-semantic / classification-driven artifacts.
    - Reduce 'watercolor / painterly GAN' bias by:
      - focusing on faithful local textures,
      - keeping loss_weight modest and combining with strong fidelity terms.

    Config (YAML):
      - type: convnextperceptualloss
        loss_weight: 0.15  # typical range: 0.1 - 0.2
        layers: [1, 2]     # optional override of feature indices
        layer_weights: [1.0, 0.5]  # optional, same length as layers

    Notes:
    - Expects inputs in [0, 1]. Internally normalized to ImageNet stats.
    - Backbone is frozen and run in eval() mode.
    - Uses L1 distance on feature maps, aggregated over selected layers.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        layers: list[int] | None = None,
        layer_weights: list[float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

        # Which feature indices from timm's features_only to use.
        # Default: low/mid-level (1 and 2).
        if layers is None:
            layers = [1, 2]
        self.layers = list(layers)

        # Relative weights per selected layer.
        if layer_weights is None:
            # Default: emphasize shallower more than deeper: [1.0, 0.5]
            layer_weights = (
                [1.0, 0.5] if len(self.layers) == 2 else [1.0] * len(self.layers)
            )
        if len(layer_weights) != len(self.layers):
            raise ValueError(
                f"ConvNeXtPerceptualLoss: layer_weights (len={len(layer_weights)}) "
                f"must match layers (len={len(self.layers)})."
            )
        self.layer_weights = nn.Parameter(
            torch.tensor(layer_weights, dtype=torch.float32),
            requires_grad=False,
        )

        # Backbone: ConvNeXt-Tiny feature extractor (frozen).
        backbone = _create_convnext_tiny_features()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self.backbone = backbone

        # Register normalization buffers for ImageNet stats.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @torch.no_grad()
    def _extract_feats(self, x: Tensor) -> list[Tensor]:
        # x: [B, 3, H, W] in [0, 1]
        # Clamp and normalize to ImageNet stats.
        x = x.clamp(0.0, 1.0)

        # Work around Pylance/pyright union typing on registered buffers by casting.
        mean: Tensor = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std: Tensor = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)

        x = (x - mean) / (std + self.eps)

        # timm features_only backbones return a list/tuple of feature maps.
        feats_out = self.backbone(x)  # type: ignore[call-arg]
        if isinstance(feats_out, (list, tuple)):
            feats: list[Tensor] = list(feats_out)
        else:  # pragma: no cover - defensive; timm should always return list/tuple here.
            raise TypeError(
                f"ConvNeXtPerceptualLoss: expected list/tuple from backbone, got {type(feats_out)}"
            )

        selected: list[Tensor] = []
        for idx in self.layers:
            if idx < 0 or idx >= len(feats):
                raise IndexError(
                    f"ConvNeXtPerceptualLoss: requested layer index {idx} "
                    f"out of range for features length {len(feats)}."
                )
            selected.append(feats[idx])
        return selected

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage]
    def forward(self, x: Tensor, gt: Tensor) -> dict[str, Tensor]:
        """Compute ConvNeXt-Tiny based perceptual loss.

        Returns dict with key 'convnext_perceptual' to integrate with existing loss handling.
        """
        # Ensure channel-last AMP conversions upstream don't break shapes.
        if x.ndim != 4 or gt.ndim != 4:
            raise ValueError(
                "ConvNeXtPerceptualLoss expects inputs of shape [B, C, H, W]."
            )

        with torch.no_grad():
            feats_gt = self._extract_feats(gt)

        feats_x = self._extract_feats(x)

        loss: Tensor = x.new_tensor(0.0)
        for w, fx, fy in zip(self.layer_weights, feats_x, feats_gt, strict=False):
            if fx.shape != fy.shape:
                # Safety: interpolate fy to fx resolution if there is tiny mismatch.
                if fx.shape[2:] != fy.shape[2:]:
                    fy = F.interpolate(
                        fy, size=fx.shape[2:], mode="bilinear", align_corners=False
                    )
            l1 = F.l1_loss(fx, fy)
            loss = loss + w * l1

        # Normalize by sum of weights to keep scale predictable.
        weight_sum = float(self.layer_weights.sum().item())
        if weight_sum > 0:
            loss = loss / weight_sum

        loss = loss * self.loss_weight

        return {"convnext_perceptual": loss}
