# ParagonSR2 Release Bundle

This release includes a complete super-resolution training and deployment toolkit:

## 📦 Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **ParagonSR2** | Dual-Path SISR Generator | [README_ParagonSR2.md](README_ParagonSR2.md) |
| **MUNet** | Multi-Branch U-Net Discriminator | [README_MUNet.md](README_MUNet.md) |
| **FeatureMatchingLoss** | GAN training stabilization loss | See MUNet docs |
| **convert_onnx_release.py** | ONNX/TensorRT export script | Usage below |

---

## ⚡ Quick Deployment

### Step 1: Export to ONNX
```bash
python convert_onnx_release.py \
    --checkpoint "paragonsr2_photo_x2.safetensors" \
    --arch paragonsr2_photo \
    --scale 2 \
    --output "release_onnx"
```

### Step 2: Build TensorRT Engine
```bash
trtexec --onnx=release_onnx/paragonsr2_photo_fp32.onnx \
        --saveEngine=paragonsr2_photo_fp16.trt \
        --fp16 \
        --minShapes=input:1x3x64x64 \
        --optShapes=input:1x3x720x1280 \
        --maxShapes=input:1x3x1080x1920
```

---

## 🛠️ Recommended Training Setup

```yaml
# Generator
network_g:
  type: paragonsr2_photo
  scale: 2
  upsampler_alpha: 0.0  # For fidelity training

# Discriminator
network_d:
  type: MUNet
  num_feat: 64
  ch_mult: [1, 2, 4, 8]

# Losses
losses:
  - type: L1Loss
    loss_weight: 1.0
  - type: PerceptualLoss
    loss_weight: 0.1
  - type: GANLoss
    loss_weight: 0.1
  - type: FeatureMatchingLoss
    loss_weight: 1.0
```

---

## 📜 License

MIT License - Philip Hofmann
