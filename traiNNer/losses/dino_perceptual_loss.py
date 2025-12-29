from collections.abc import Sequence
from typing import Any

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DINOPerceptualLoss(nn.Module):
    """
    A feature-based perceptual loss using self-supervised DINO / DINOv2 / DINOv3 ViT encoders from timm.

    Key features:
    - Robust against different timm forward output types: dict, list, tuple, tensor
    - Supports features_only models and fallback models
    - Works with token sequences or 4D spatial feature maps
    - Supports layer selection by name or by index or ["last"]
    - Charbonnier distance with optional weights per layer
    - Optional flexible resizing to maintain patch-grid shapes
    - Optional debug logging

    Example config:
        perceptual_loss:
            type: DINOPerceptualLoss
            loss_weight: 0.18
            model_name: vit_small_patch16_dinov3
            layers: ['feat2','last']
            weights: [1.0, 0.5]
            resize: true
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        model_name: str = "vit_small_patch16_dinov3",
        layers: Sequence[str | int] = ("last",),
        weights: Sequence[float] | None = None,
        resize: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.model_name = model_name
        self.layers = list(layers)
        self.weights = [1.0] * len(layers) if weights is None else list(weights)
        self.flexible_resize = resize
        self.debug = debug

        # Register normalization buffers
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.eps = 1e-6

        self.backbone = self._create_backbone(model_name)

        # Dynamic patch size detection
        self.patch_size = self._detect_patch_size()

        # Validate weights configuration
        self._validate_config()

        self.charbonnier = lambda x: torch.sqrt(x + self.eps**2)

    # -----------------------------
    # Backbone loading
    # -----------------------------
    def _create_backbone(self, name: str) -> nn.Module:
        model = None
        try:
            model = timm.create_model(name, pretrained=True, features_only=True)
        except Exception:
            model = timm.create_model(
                name, pretrained=True, num_classes=0, global_pool=""
            )

        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

    # -----------------------------
    # Robust feature extraction
    # -----------------------------
    @torch.no_grad()
    def _extract_feats(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError("Expected [B,C,H,W]")

        orig_size = x.shape[2:]
        x = x.clamp(0, 1)

        if self.flexible_resize:
            target_size = self._find_optimal_size(*orig_size)
            if target_size != orig_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

        # Use registered buffers with proper dtype and device handling
        mean: Tensor = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std: Tensor = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        x = (x - mean) / (std + self.eps)

        if hasattr(self.backbone, "forward_features"):
            try:
                feats_raw = self.backbone.forward_features(x)  # type: ignore[operator]
            except Exception:
                feats_raw = self.backbone(x)
        else:
            feats_raw = self.backbone(x)

        feats: dict[str, Tensor] = {}

        def tokens_to_map(t: Tensor, name: str) -> None:
            """Convert token sequences to spatial feature maps with better error handling."""
            if t.ndim != 3 or t.shape[1] <= 1:
                if self.debug:
                    print(
                        f"[DINO DEBUG] Skipping {name}: not enough tokens ({t.shape[1]}) or wrong dim ({t.ndim})"
                    )
                return

            # Remove CLS token (first token) and check for square patch grid
            patch_tokens = t[:, 1:, :]
            B, N, C = patch_tokens.shape

            if N <= 0:
                if self.debug:
                    print(f"[DINO DEBUG] Skipping {name}: no patch tokens")
                return

            # Check if token count forms a perfect square (valid patch grid)
            h = int(N**0.5)
            if h * h != N:
                if self.debug:
                    print(
                        f"[DINO DEBUG] Skipping {name}: {N} tokens don't form perfect square grid ({h}x{h})"
                    )
                return

            try:
                fmap = patch_tokens.transpose(1, 2).reshape(B, C, h, h)
                feats[name] = fmap
                if self.debug:
                    print(
                        f"[DINO DEBUG] Successfully converted {name}: {B}x{C}x{h}x{h}"
                    )
            except Exception as e:
                if self.debug:
                    print(f"[DINO DEBUG] Failed to convert {name} to spatial map: {e}")

        if isinstance(feats_raw, (list, tuple)):
            for idx, t in enumerate(feats_raw):
                if not isinstance(t, Tensor):
                    continue
                if t.ndim == 4:
                    feats[f"feat{idx}"] = t
                elif t.ndim == 3:
                    tokens_to_map(t, f"feat{idx}")

        elif isinstance(feats_raw, dict):
            for k, v in feats_raw.items():
                if not isinstance(v, Tensor):
                    continue
                if v.ndim == 4:
                    feats[k] = v
                elif v.ndim == 3:
                    tokens_to_map(v, k)

        elif isinstance(feats_raw, Tensor):
            if feats_raw.ndim == 4:
                feats["feat0"] = feats_raw
            elif feats_raw.ndim == 3:
                tokens_to_map(feats_raw, "feat0")

        if not feats:
            raise RuntimeError(
                f"DINOPerceptualLoss: no usable features in model '{self.model_name}'."
            )

        if self.debug:
            print(f"[DINO DEBUG] Features extracted: {list(feats.keys())}")
            for name, tensor in feats.items():
                print(f"[DINO DEBUG]   {name}: {tensor.shape}")

        return feats

    def get_available_layers(self) -> list[str]:
        """Get list of available feature layer names for this model.

        This is useful for debugging and configuration.
        """
        # Create a dummy input to get feature shapes
        dummy_input = torch.randn(1, 3, 224, 224)  # Standard ViT input size
        with torch.no_grad():
            try:
                feats = self._extract_feats(dummy_input)
                return list(feats.keys())
            except Exception as e:
                if self.debug:
                    print(f"[DINO DEBUG] Failed to get available layers: {e}")
                return []

    def forward(self, x: Tensor, gt: Tensor) -> dict[str, Tensor]:
        """Compute DINO perceptual loss.

        Returns dict with key 'dino_perceptual' to integrate with existing loss handling.
        """
        if x.ndim != 4 or gt.ndim != 4:
            raise ValueError("DINOPerceptualLoss expects inputs of shape [B, C, H, W].")

        x_feats = self._extract_feats(x)
        y_feats = self._extract_feats(gt)

        # Initialize as tensor to ensure consistent typing
        total = x.new_tensor(0.0)
        for layer, w in zip(self.layers, self.weights, strict=False):
            fmap_x = self._get_layer(x_feats, layer)
            fmap_y = self._get_layer(y_feats, layer)
            diff = self.charbonnier((fmap_x - fmap_y) ** 2).mean()
            total += w * diff

        # Normalize by sum of weights to keep scale predictable
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            total = total / weight_sum

        total = total * self.loss_weight

        return {"dino_perceptual": total}

    def _get_layer(self, feats: dict[str, Tensor], layer: str | int) -> Tensor:
        """More precise layer selection with better error handling."""
        keys = list(feats.keys())

        if layer == "last":
            return feats[keys[-1]]

        if isinstance(layer, int):
            layer = f"feat{layer}"
            if layer in feats:
                return feats[layer]
            else:
                raise KeyError(
                    f"Layer index {layer} not found in available layers: {keys}"
                )

        # Exact match first
        if layer in feats:
            return feats[layer]

        # More precise partial matching - require exact feature name match
        import re

        # Match patterns like "feat1", "layer1", etc. but not "feat11" for "feat1"
        pattern = r"\b" + re.escape(layer) + r"\b"
        matches = [k for k in keys if re.search(pattern, k)]

        if len(matches) == 1:
            if self.debug:
                print(f"[DINO DEBUG] Fuzzy matched layer '{layer}' to '{matches[0]}'")
            return feats[matches[0]]
        elif len(matches) > 1:
            raise ValueError(
                f"Layer '{layer}' matches multiple keys: {matches}. Be more specific."
            )
        else:
            # Fallback to last layer with warning
            import warnings

            warnings.warn(
                f"Layer '{layer}' not found in available layers: {keys}. Using last layer instead.",
                UserWarning,
                stacklevel=2,
            )
            return feats[keys[-1]]

    def _detect_patch_size(self) -> int:
        """Automatically detect patch size from the backbone model."""
        # Common patch sizes for different model families
        if hasattr(self.backbone, "patch_size"):
            return self.backbone.patch_size
        elif hasattr(self.backbone, "patch_embed"):
            if hasattr(self.backbone.patch_embed, "patch_size"):
                return (
                    self.backbone.patch_embed.patch_size[0]
                    if isinstance(self.backbone.patch_embed.patch_size, tuple)
                    else self.backbone.patch_embed.patch_size
                )

        # Fallback based on model name patterns (more robust than hardcoding)
        model_name_lower = self.model_name.lower()
        if "patch8" in model_name_lower or "vit_tiny" in model_name_lower:
            return 8
        elif (
            "patch16" in model_name_lower
            or "vit_small" in model_name_lower
            or "dinov2" in model_name_lower
        ):
            return 16
        elif "patch32" in model_name_lower or "vit_base" in model_name_lower:
            return 32
        else:
            # Conservative fallback - likely to work with most models
            if self.debug:
                print(
                    "[DINO DEBUG] Patch size detection failed, using conservative fallback: 16"
                )
            return 16

    def _validate_config(self) -> None:
        """Validate configuration and warn about potential issues."""
        if len(self.weights) != len(self.layers):
            import warnings

            warnings.warn(
                f"DINOPerceptualLoss: layers ({len(self.layers)}) and weights ({len(self.weights)}) length mismatch. "
                f"Using weight=1.0 for missing entries.",
                UserWarning,
                stacklevel=2,
            )
            # Extend weights to match layers with default weight of 1.0
            self.weights.extend([1.0] * (len(self.layers) - len(self.weights)))

    # -----------------------------
    # Patch size helper
    # -----------------------------
    def _find_optimal_size(self, h: int, w: int) -> tuple[int, int]:
        patch = self.patch_size
        h_new = (h // patch) * patch
        w_new = (w // patch) * patch
        return h_new if h_new > 0 else patch, w_new if w_new > 0 else patch
