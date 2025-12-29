# New Loss Functions for traiNNer-redux

This directory contains two new loss functions designed to replace unstable GAN frameworks with more stable and semantically-aware alternatives:

## 1. ContrastiveLoss (`contrastive_loss.py`)

A CLIP-based contrastive loss that encourages semantic similarity between super-resolved and ground truth images while pushing away negative samples.

### Features:
- Uses OpenAI's CLIP model for semantic feature extraction
- Implements InfoNCE-style contrastive loss
- Creates negative samples using bicubic upsampling of low-quality inputs
- Gracefully falls back to a simplified loss if CLIP is not available

### Requirements:
- `transformers` library (automatically installed with the updated `pyproject.toml`)

### Usage:
```yaml
- type: contrastiveloss
  loss_weight: 0.1
  temperature: 0.1
```

## 2. LaplacianPyramidLoss (`laplacian_loss.py`)

A multi-scale loss that preserves structural details by comparing Laplacian pyramids of the super-resolved and ground truth images.

### Features:
- Builds Gaussian and Laplacian pyramids at multiple scales
- Supports L1, L2, and Charbonnier loss criteria
- Preserves both fine and coarse details across different scales

### Usage:
```yaml
- type: laplacianpyramidloss
  loss_weight: 1.0
  levels: 4
  criterion: charbonnier
```

## Configuration Example

To replace a GAN-based training configuration with these new losses:

1. Remove `network_d` and `optim_d` sections
2. Remove `ganloss` from the losses list
3. Add the new losses as shown above
4. Replace `MultiStepLR` scheduler with `CosineAnnealingLR`:

```yaml
scheduler:
  type: CosineAnnealingLR
  T_max: 1000000
  eta_min: !!float 1e-7
```

This approach provides more stable training with better perceptual quality while avoiding the common artifacts associated with GAN-based approaches.
