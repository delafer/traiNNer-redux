from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as v2F

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ContrastiveLoss(nn.Module):
    """CLIP-based Contrastive Loss for Super-Resolution.

    This loss uses CLIP features to encourage the super-resolved image to be
    semantically similar to the ground truth while being dissimilar to a negative sample.
    """

    def __init__(self, loss_weight: float = 0.1, temperature: float = 0.1) -> None:
        """
        Initialize the ContrastiveLoss.

        Args:
            loss_weight (float): Weight for the loss. Default: 0.1
            temperature (float): Temperature parameter for the contrastive loss. Default: 0.1
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be a positive float.")

        self.loss_weight = float(loss_weight)
        self.temperature = float(temperature)

        # CLIP integration state
        self.use_clip = False
        self.clip_model: nn.Module | None = None
        self.clip_preprocess = v2.Compose(
            [
                v2.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
                v2.CenterCrop(224),
                v2.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self._clip_model_name = "openai/clip-vit-base-patch32"
        self._clip_device: torch.device | None = None
        self._clip_cls: Any = None

        try:
            from transformers import CLIPModel  # type: ignore

            self._clip_cls = CLIPModel
            self.use_clip = True
        except ImportError:
            print(
                "Warning: transformers library not found. Using simplified contrastive loss."
            )
        except Exception as exc:
            print(
                f"Warning: Could not provision CLIP model ({exc}). Using simplified contrastive loss."
            )
            self._clip_cls = None
            self.use_clip = False

    def _ensure_clip_model(self, device: torch.device) -> bool:
        """
        Lazily load and place the CLIP model on the requested device.

        Returns:
            bool: True if the CLIP model is available and ready, False otherwise.
        """
        clip_cls = self._clip_cls
        if not self.use_clip or clip_cls is None:
            return False

        clip_model = self.clip_model
        if clip_model is None:
            try:
                clip_model = cast(
                    nn.Module,
                    clip_cls.from_pretrained(self._clip_model_name),  # type: ignore[attr-defined]
                )
            except Exception as exc:  # pragma: no cover - download/runtime error
                print(
                    f"Warning: Failed to load CLIP model '{self._clip_model_name}': {exc}. "
                    "Falling back to simplified contrastive loss."
                )
                self.clip_model = None
                self.use_clip = False
                return False

            for param in clip_model.parameters():
                param.requires_grad = False
            clip_model.eval()
            self.clip_model = clip_model

        if self._clip_device != device:
            clip_model = self.clip_model
            assert clip_model is not None
            clip_model = clip_model.to(device)
            self.clip_model = clip_model
            self._clip_device = device

        return True

    def extract_clip_features(self, images: torch.Tensor) -> torch.Tensor | None:
        """Extract features from images using CLIP."""
        if not self._ensure_clip_model(images.device):
            return None

        images = self.clip_preprocess(images).float()

        clip_model = self.clip_model
        assert clip_model is not None
        with torch.no_grad():
            outputs = clip_model.get_image_features(pixel_values=images)  # type: ignore[arg-type]

        return outputs

    def forward(
        self, sr: torch.Tensor, gt: torch.Tensor, lq: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for contrastive loss.

        Args:
            sr (torch.Tensor): Super-resolved images of shape (N, C, H, W)
            gt (torch.Tensor): Ground truth images of shape (N, C, H, W)
            lq (torch.Tensor): Low-quality images of shape (N, C, H, W)

        Returns:
            torch.Tensor: Computed contrastive loss
        """
        # Create negative samples using bicubic upsampling
        target_size = (gt.shape[2], gt.shape[3])
        negative = v2F.resize(
            lq,
            size=list(target_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        if self.use_clip:
            # Extract CLIP features
            sr_features = self.extract_clip_features(sr)
            gt_features = self.extract_clip_features(gt)
            negative_features = self.extract_clip_features(negative)

            if (
                sr_features is not None
                and gt_features is not None
                and negative_features is not None
            ):
                # Normalize features
                sr_features = F.normalize(sr_features, dim=-1)
                gt_features = F.normalize(gt_features, dim=-1)
                negative_features = F.normalize(negative_features, dim=-1)

                # Compute cosine similarities
                pos_sim = torch.sum(sr_features * gt_features, dim=-1)  # Shape: (N,)
                neg_sim = torch.sum(
                    sr_features * negative_features, dim=-1
                )  # Shape: (N,)

                # Compute InfoNCE loss
                logits = torch.stack([pos_sim, neg_sim], dim=1) / self.temperature
                labels = torch.zeros(
                    logits.shape[0], dtype=torch.long, device=logits.device
                )

                loss = F.cross_entropy(logits, labels, reduction="mean")

                return self.loss_weight * loss

        # Fallback to simplified loss if CLIP is not available
        # Using L1 distance as a substitute for feature-based similarity
        pos_sim = -F.l1_loss(sr, gt, reduction="none").mean(dim=[1, 2, 3])
        neg_sim = -F.l1_loss(sr, negative, reduction="none").mean(dim=[1, 2, 3])

        # Simple contrastive loss (InfoNCE-style)
        logits = torch.stack([pos_sim, neg_sim], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels, reduction="mean")

        return self.loss_weight * loss
