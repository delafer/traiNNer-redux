from __future__ import annotations

from collections.abc import Iterable
from typing import Any, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "timm is required for DINOPerceptualLoss. "
        "Please install timm or remove dinoperceptualloss from your config."
    ) from exc


def _create_dino_backbone(model_name: str, pretrained: bool = True) -> nn.Module:
    """Create a DINO/DINOv2 ViT backbone for perceptual features.

    Uses timm to load a vision transformer with DINO-style pretraining.
    The model is configured for feature extraction (no classifier / head).

    Args:
        model_name: timm model name, e.g. "vit_small_patch14_dinov2.lvd142m".
        pretrained: if True, load pretrained weights.

    Returns:
        Frozen nn.Module in eval mode.
    """
    # Configure model for feature extraction:
    # - num_classes=0, global_pool="" prevents creating an unwanted head in many timm models.
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
    )

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    return model


@LOSS_REGISTRY.register()
class DINOPerceptualLoss(nn.Module):
    """DINOv2-based perceptual loss for SISR.

    Design goals:
    - Modern alternative to VGG/ConvNeXt perceptual losses.
    - Emphasize low/mid-level structure and textures.
    - Fully frozen backbone; no gradients through DINO.
    - AMP-safe; loss used only at train-time.

    YAML example:

      - type: dinoperceptualloss
        loss_weight: 0.18
        model_name: vit_small_patch14_dinov2.lvd142m
        layers: ["block3", "block6", "block9"]
        layer_weights: [1.0, 0.7, 0.5]
        use_charbonnier: true

    Notes:
    - Expects inputs in [0, 1].
    - Internally normalized to ImageNet mean/std; adjust if using custom DINO stats.
    - We match configured layer names as substrings of available feature keys to stay
      robust across timm versions; if not found, we fall back to the last feature map.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        layers: list[str] | None = None,
        layer_weights: list[float] | None = None,
        use_charbonnier: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        self.use_charbonnier = bool(use_charbonnier)

        # Logical layer identifiers we will match against extracted feature keys.
        if layers is None:
            layers = ["block3", "block6", "block9"]
        self.layers = list(layers)

        if layer_weights is None:
            # Decreasing weights for deeper features by default.
            if len(self.layers) == 1:
                layer_weights = [1.0]
            else:
                layer_weights = [1.0] + [0.7] * (len(self.layers) - 1)
        if len(layer_weights) != len(self.layers):
            raise ValueError(
                f"DINOPerceptualLoss: layer_weights (len={len(layer_weights)}) "
                f"must match layers (len={len(self.layers)})."
            )

        self.register_buffer(
            "layer_weights",
            torch.tensor(layer_weights, dtype=torch.float32),
            persistent=False,
        )

        # Backbone: DINO/DINOv2 ViT from timm.
        self.backbone = _create_dino_backbone(model_name=model_name, pretrained=True)

        # Normalization buffers (ImageNet-style)
        # If you want DINOv2-specific stats, update here accordingly.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @torch.no_grad()
    def _extract_feats(self, x: Tensor) -> dict[str, Tensor]:
        """Extract DINO features as spatial maps.

        Strategy:
        - Normalize input.
        - Call backbone.forward_features(x_norm).
        - Interpret outputs:
          * If dict: use all tensor entries; convert [B, N, C] tokens to maps.
          * If tensor [B, N, C]: convert patch tokens to map.
          * If tensor [B, C, H, W]: treat directly as a feature map.
        - Store as {str_key: [B, C, H, W]}.

        Returns:
            dict mapping feature names to 4D feature maps.

        Raises:
            RuntimeError if we cannot derive any usable feature maps.
        """
        if x.ndim != 4:
            raise ValueError("DINOPerceptualLoss expects inputs of shape [B, C, H, W].")

        # Clamp and normalize
        x = x.clamp(0.0, 1.0)

        mean: Tensor = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std: Tensor = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)

        x_norm = (x - mean) / (std + self.eps)

        # forward_features is the canonical timm API for ViT-like models.
        # Some models may return dict, some tensor.
        feats_raw: Any = self.backbone.forward_features(x_norm)  # type: ignore[operator]

        feats: dict[str, Tensor] = {}

        # Helper to convert token sequences to spatial maps
        def tokens_to_map(tokens: Tensor, key_prefix: str) -> None:
            # tokens: [B, N, C] possibly with CLS token at index 0
            if tokens.ndim != 3 or tokens.shape[1] <= 4:
                return
            # Assume first token is CLS; drop it
            patch_tokens = tokens[:, 1:, :]
            b, n, c = patch_tokens.shape
            h = int(n**0.5)
            w = h
            if h * w != n:
                return
            fmap = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
            feats[f"{key_prefix}"] = fmap

        if isinstance(feats_raw, dict):
            # Use all tensor entries
            for k, v in feats_raw.items():
                if not isinstance(v, Tensor):
                    continue
                # Token sequence
                if v.ndim == 3:
                    tokens_to_map(v, k)
                # Already spatial
                elif v.ndim == 4:
                    feats[k] = v
        elif isinstance(feats_raw, Tensor):
            v = feats_raw
            if v.ndim == 3:
                tokens_to_map(v, "block_last")
            elif v.ndim == 4:
                feats["block_last"] = v

        if not feats:
            raise RuntimeError(
                "DINOPerceptualLoss: unable to extract features from DINO backbone. "
                "Check model_name compatibility or update extractor logic."
            )

        return feats

    def _feature_loss(self, fx: Tensor, fy: Tensor) -> Tensor:
        """Compute per-location feature discrepancy."""
        if fx.shape != fy.shape:
            fy = F.interpolate(
                fy,
                size=fx.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.use_charbonnier:
            diff = fx - fy
            return torch.sqrt(diff * diff + self.eps)
        return torch.abs(fx - fy)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage]
    def forward(self, x: Tensor, gt: Tensor) -> dict[str, Tensor]:
        """Compute DINO-based perceptual loss.

        Returns:
            dict with key "dino_perceptual" for integration with generic loss handler.
        """
        # Extract GT feats without grad
        with torch.no_grad():
            feats_gt_all = self._extract_feats(gt)

        feats_x_all = self._extract_feats(x)

        loss = x.new_tensor(0.0)

        # For each configured layer key, match by substring; fall back to last map.
        # self.layer_weights is a 1D buffer tensor; convert to a Python list for clean typing.
        weights: list[float] = [float(v) for v in self.layer_weights]  # type: ignore[union-attr]

        for w, layer_key in zip(weights, self.layers, strict=False):
            # gather candidate matches
            matched_x: list[Tensor] = [
                v for k, v in feats_x_all.items() if layer_key in k
            ]
            matched_gt: list[Tensor] = [
                v for k, v in feats_gt_all.items() if layer_key in k
            ]

            if matched_x and matched_gt:
                fx = matched_x[0]
                fy = matched_gt[0]
            else:
                # Fallback: last inserted feature map to keep behavior defined
                fx = next(reversed(feats_x_all.values()))
                fy = next(reversed(feats_gt_all.values()))

            fl = self._feature_loss(fx, fy).mean()
            loss = loss + w * fl

        weight_sum = float(self.layer_weights.sum().item())
        if weight_sum > 0:
            loss = loss / weight_sum

        loss = loss * self.loss_weight

        return {"dino_perceptual": loss}
