import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import InterpolationMode, v2

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
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.use_clip = False

        try:
            from transformers import CLIPModel

            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

            # Freeze CLIP model
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()

            # Preprocessing transforms
            self.clip_preprocess = v2.Compose(
                [
                    v2.Resize(
                        224, interpolation=InterpolationMode.BICUBIC, antialias=True
                    ),
                    v2.CenterCrop(224),
                    v2.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
            self.use_clip = True

        except ImportError:
            print(
                "Warning: transformers library not found. Using simplified contrastive loss."
            )
        except Exception as e:
            print(
                f"Warning: Could not load CLIP model: {e}. Using simplified contrastive loss."
            )

    def extract_clip_features(self, images: torch.Tensor) -> torch.Tensor | None:
        """Extract features from images using CLIP."""
        if not self.use_clip:
            return None

        # Move CLIP model to the same device as the input images
        self.clip_model.to(images.device)

        # Preprocess images
        images = self.clip_preprocess(images)

        # Extract features
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(pixel_values=images)

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
        bicubic_upscale = v2.Resize(
            (gt.shape[2], gt.shape[3]),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        negative = bicubic_upscale(lq)

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
