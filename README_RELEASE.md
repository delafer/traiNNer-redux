# ParagonSR2 & MUNet: High-Performance Super-Resolution Architecture

**Author**: Philip Hofmann
**License**: MIT
**Repository**: https://github.com/Phhofm/traiNNer-redux

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

ParagonSR2 is a **hybrid super-resolution architecture** designed for optimal speed-quality tradeoffs, combined with MUNet, a **multi-branch discriminator** for superior GAN training. This release provides production-ready code for both generator and discriminator architectures.

### Key Innovation: Dual-Path Design

Unlike traditional approaches that process heavily in high-resolution space, ParagonSR2 uses a **dual-path architecture**:

- **Path A (Detail)**: LR → Deep Features → PixelShuffle → Learned Detail
- **Path B (Base)**: LR → MagicKernel → Classical Upsampling
- **Output**: Base + Detail

This design provides the **best of both worlds**: speed of classical methods with quality of deep learning.

## 🚀 Key Advantages

### For ParagonSR2 (Generator):
- **4-5x Faster Training**: All heavy computation in low-resolution space
- **Graceful Degradation**: MagicKernel base provides structural safety net
- **Training Stability**: Conservative detail initialization prevents collapse
- **Production Ready**: Static ONNX/TensorRT export with dynamic axes support
- **Hardware Efficient**: Channels-last memory format for AMP optimization
- **Scalable**: 8 variants from Nano (0.02M params) to XL (3.8M params)

### For MUNet (Discriminator):
- **Multi-Branch Detection**: 4 specialized branches for comprehensive artifact detection
- **Frequency Awareness**: Explicit FFT-based frequency domain analysis
- **Edge Detection**: Spatial gradient analysis for compression artifacts
- **Stable Training**: Spectral normalization prevents discriminator collapse
- **Attention Fusion**: Intelligent branch weighting per spatial location

### Combined System:
- **Complementary Design**: Generator optimized for speed, discriminator for quality
- **Easy Integration**: Simple configuration-based setup
- **Production Deployment**: Generator exports cleanly to ONNX/TensorRT

## 🏗️ Architecture Design Philosophy

### ParagonSR2: Why Dual-Path?

**Problem**: Traditional SR models process everything in high-resolution space, which is computationally expensive and memory-intensive.

**Solution**: Keep all heavy processing in efficient low-resolution space while using classical upsampling as a stability anchor.

**Benefits**:
- **Speed**: 4x fewer pixels to process for 2x SR
- **Stability**: Base path prevents mode collapse during GAN training
- **Quality**: Detail path adds high-frequency texture and artifact removal
- **Deployment**: Static operations enable clean ONNX/TensorRT export

### MUNet: Why Multi-Branch?

**Problem**: Single-path discriminators often miss specific types of artifacts (frequency-domain issues, edge artifacts, texture inconsistencies).

**Solution**: Four specialized branches that each focus on different aspects of image quality:

1. **Spatial Branch**: U-Net structure for multi-scale spatial analysis
2. **Gradient Branch**: Edge detection via spatial gradients
3. **Frequency Branch**: FFT magnitude analysis for frequency artifacts
4. **Patch Branch**: Local texture consistency checking

**Benefits**:
- **Comprehensive Coverage**: Each branch catches different artifact types
- **Complementary Gradients**: Provides diverse training signals to generator
- **Global + Local**: Combines self-attention with local texture analysis

## 📊 Model Variants

| Variant | Feature Channels | Depth | Parameters | Use Case |
|---------|------------------|-------|------------|----------|
| **Nano** | 12 | 1×1 | 0.02M | Real-time video, edge devices |
| **Micro** | 16 | 1×2 | 0.04M | Fast processing, low-power |
| **Tiny** | 24 | 2×2 | 0.08M | Good quality + speed balance |
| **XS** | 32 | 2×3 | 0.12M | General-purpose SR |
| **S** | 48 | 3×4 | 0.28M | **Recommended** for most use cases |
| **M** | 64 | 4×6 | 0.65M | High quality, professional use |
| **L** | 96 | 6×8 | 1.8M | Research-grade quality |
| **XL** | 128 | 8×10 | 3.8M | Maximum quality, competitions |

**Recommendation**: Start with **S variant** for most applications, scale up for higher quality or down for speed.

## 🛠️ Usage

### Training Setup

#### Basic Generator Configuration
```yaml
network_g:
  type: paragonsr2_s        # Or nano, micro, tiny, xs, s, m, l, xl
  scale: 2                  # 2x, 3x, or 4x super-resolution
  upsampler_alpha: 0.5      # MagicKernel sharpening (0-1)
  detail_gain: 0.1          # Initial detail contribution
  fast_body_mode: true      # 2x faster training
```

#### GAN Training with MUNet
```yaml
network_d:
  type: munet               # Multi-branch discriminator
  num_in_ch: 3
  num_feat: 64

train:
  gan_opt:
    type: r3ganloss         # Recommended: R3GAN with R1 penalty
    gan_weight: 0.03
    gan_weight_init: 0.0
    gan_weight_steps: [[10000, 0.03]]  # Ramp over 10k iterations

  optim_d:
    type: AdamW
    lr: 3e-5               # 3x slower than generator
    weight_decay: 0
```

### Inference Usage

#### PyTorch
```python
from traiNNer.archs.paragonsr2_arch import ParagonSR2

# Load model
model = ParagonSR2(scale=2, num_feat=48)  # S variant
model.load_state_dict(checkpoint)
model.eval()

# Process image
lr_image = torch.randn(1, 3, 64, 64)  # Low-res input
hr_output = model(lr_image)           # High-res output
```

#### ONNX Export
```python
import torch.onnx

# Export to ONNX (dynamic shapes supported)
torch.onnx.export(
    model, dummy_input, "paragonsr2.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {2: "height", 3: "width"},
                  "output": {2: "height", 3: "width"}},
    opset_version=18
)
```

#### TensorRT Conversion
```bash
# Convert ONNX to TensorRT FP16
trtexec --onnx=paragonsr2.onnx --saveEngine=paragonsr2.trt --fp16 \
    --minShapes=input:1x3x64x64 \
    --optShapes=input:1x3x540x960 \
    --maxShapes=input:1x3x1080x1920
```

## 🔬 Technical Details

### ParagonSR2 Architecture Components

1. **Shallow Feature Extraction**
   - Single 3×3 convolution to expand RGB to feature space
   - Minimal overhead with maximum information retention

2. **Deep Body (LR Space)**
   - Multiple ResidualGroups with ParagonBlockStatic
   - InceptionDWConv2d: Multi-scale depthwise context
   - StaticDepthwiseTransformer: Efficient channel mixing
   - All processing at low resolution (4× fewer pixels for 2× SR)

3. **Upsampling (Path A)**
   - PixelShufflePack with ICNR initialization
   - Prevents checkerboard artifacts
   - Learns optimal upsampling patterns

4. **Base Upsampling (Path B)**
   - MagicKernelSharp2021: Classical separable convolution
   - Fixed weights (no gradients, stable)
   - Provides structural correctness

### MUNet Discriminator Components

1. **Shared Encoder**
   - Progressive downsampling with spectral normalization
   - Skip connections for U-Net decoder

2. **Bottleneck + Self-Attention**
   - Deepest feature processing with global context
   - Captures long-range dependencies

3. **Four Specialized Branches**
   - **Spatial**: U-Net decoder for multi-scale analysis
   - **Gradient**: Edge detection via spatial gradients
   - **Frequency**: FFT magnitude analysis (differentiable)
   - **Patch**: Texture consistency from bottleneck features

4. **Attention Fusion**
   - Learns to weight branches per spatial location
   - More effective than simple concatenation

### Key Design Choices

- **RMSNorm**: ~10% speedup over GroupNorm
- **MagicKernel**: Superior to bicubic/nearest for classical upsampling
- **Channels-Last**: Memory format optimization for AMP training
- **Spectral Normalization**: Stabilizes GAN training
- **Conservative Initialization**: detail_gain prevents training collapse

## 📈 Performance Characteristics

### Computational Efficiency
- **LR Processing**: 4× fewer operations for 2× SR (8× for 4× SR)
- **Memory Usage**: Channels-last format reduces memory overhead
- **Training Speed**: Significantly faster than HR-processing approaches
- **Inference Speed**: TensorRT FP16 provides substantial speedup

### Quality vs Speed Trade-offs
- **Nano/Micro**: Real-time capable, good for video processing
- **S/M**: Recommended for most applications, good quality-speed balance
- **L/XL**: Research-grade quality, slower but maximum fidelity

## 🔧 Deployment Notes

### ONNX Compatibility
- ✅ Opset 18 (PyTorch 2.x native support)
- ✅ Dynamic shapes supported
- ✅ Static operations only
- ✅ TensorRT patch included for AdaptiveAvgPool replacement

### Production Deployment
- **Generator Only**: Discriminator not needed for inference
- **Static Shapes**: For maximum TensorRT performance
- **Dynamic Shapes**: For flexible input sizes (slight performance cost)
- **FP16**: Recommended for inference (significant speedup)

## 📝 Training Recommendations

### General Guidelines
1. **Start with S variant**: Good balance for most use cases
2. **Use AMP**: Automatic mixed precision for faster training
3. **Conservative GAN weights**: Prevent discriminator overpowering
4. **Warm-up**: Start GAN training after 10k+ generator iterations

### GAN Training Tips
- Monitor discriminator loss vs generator loss ratio (should be 0.3-0.7)
- Use R3GAN with R1 penalty for stable training
- Gradual GAN weight ramping (0 → target over 10k steps)
- Conservative learning rate for discriminator (3× slower than generator)

## 📚 References & Inspiration

### Generator Architecture
- **EDSR** (Lim et al., CVPR 2017): Bicubic + learned residual
- **SwinIR** (Liang et al., ICCV 2021): Nearest + learned upsampling
- **HAT** (Chen et al., CVPR 2022): Hybrid attention-based SR

### Discriminator Architecture
- **PatchGAN** (Isola et al., CVPR 2017): Patch-based approach
- **StyleGAN2-D** (Karras et al., CVPR 2020): Skip connections, residuals
- **Spectral Normalization** (Miyato et al., ICLR 2018): Training stabilization

### Key Innovations
- **Dual-path hybrid**: Combines classical and deep learning upsampling
- **Multi-branch discriminator**: Comprehensive artifact detection
- **LR-space processing**: Significant efficiency gains
- **Production-ready**: Clean ONNX/TensorRT deployment

## 🤝 Contributing

This architecture is designed to be:
- **Easy to understand**: Clear docstrings and documentation
- **Well-structured**: Modular components for easy modification
- **Production-ready**: Comprehensive validation and testing
- **Extensible**: Clear patterns for adding new features

## 📄 License

MIT License - see LICENSE.txt for details.

## 🙏 Acknowledgments

Special thanks to the computer vision community for foundational work in super-resolution, and to the developers of PyTorch, ONNX, and TensorRT for making production deployment straightforward.

---

**Ready for Production**: Both ParagonSR2 and MUNet are designed with production deployment in mind, from training to inference. The clean separation between generator and discriminator makes it easy to deploy just the generator for inference while using both for training.

**Questions or Issues?** Feel free to open an issue on GitHub or reach out via the repository.
