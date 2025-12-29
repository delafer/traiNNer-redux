# ParagonSR Perceptual Fine-Tuning Configuration Guide

## Overview

This configuration template implements **realistic workflow distribution + identity learning** for perceptual fine-tuning of ParagonSR models. It's specifically designed for scenarios like training a 4x ParagonSR_S model using a pretrained base model.

## Key Features

### 🎯 **Perfect Training Distribution**

The configuration implements your exact vision:

- **10% Clean Pass-Through**: Model learns to not touch clean input (identity learning)
- **90% Realistic Workflows**: Realistic degradation chains from real-world photography workflows
- **Balanced Workflow Distribution**: Professional and phone camera workflows with internet/social media variants

### 📊 **Workflow Distribution**

```
Training Distribution:
├── 10% Clean Pass-Through (Identity Learning)
└── 90% Realistic Workflows:
    ├── 15% Professional Camera (minimal processing)
    ├── 15% Professional → Internet
    ├── 15% Professional → Social Media
    ├── 15% Phone Camera (heavy processing)
    ├── 15% Phone → Internet
    └── 25% Phone → Social Media (most common)
```

## How It Works

### **Identity Learning (10%)**
```yaml
p_clean: 0.1  # 10% chance to pass clean images through
```
This ensures the model learns to **preserve high-quality input** without unnecessary modifications.

### **Realistic Workflows (90%)**
Each workflow simulates a complete photography pipeline:

**Professional Camera Workflows:**
- Minimal sensor noise (0.5-2 sigma)
- Professional lens blur
- High-quality JPEG compression (90-98%)
- Minimal oversharpening (1.02-1.1x)
- Optional internet/social media processing

**Phone Camera Workflows:**
- Heavy sensor noise (2-8x std dev)
- Rolling shutter effects
- Hand shake motion blur
- Heavy phone sharpening (1.2-1.6x)
- Lens distortion + chromatic aberration
- Social media compression + re-upload cycles

## Usage Instructions

### **1. Replace Architecture Name**
In the configuration file, replace:
```yaml
network_g:
  type: paragonsr  # Replace with your actual model name
```

Common replacements:
- `paragon_sr_s` - ParagonSR Small
- `paragon_sr_m` - ParagonSR Medium
- `paragon_sr_l` - ParagonSR Large
- `paragon_sr_xl` - ParagonSR XL

### **2. Set Pretrained Model Path**
```yaml
path:
  pretrain_network_g: experiments/pretrained_models/YOUR_MODEL.pth
```

### **3. Adjust Dataset Paths**
```yaml
datasets:
  train:
    dataroot_gt: ["/path/to/your/hr/images"]
    dataroot_lq: ["/path/to/your/lr/images"]
```

### **4. Configure Scale Factor**
```yaml
scale: 4  # For 4x upscaling
lq_size: 64  # 64x64 LQ patches → 256x256 GT patches
```

## Workflow Details

### **Professional → Social Media (Most Aggressive)**
```python
# Complete workflow simulation:
1. Professional capture (minimal noise)
2. User editing (oversharpening 1.05-1.3x)
3. Platform processing (distortion correction)
4. Social media compression (60-85% WebP)
5. Platform sharpening (1.1-1.4x)
6. Multiple re-upload cycles
7. Final compression damage
```

### **Phone → Social Media (Real-World Most Common)**
```python
# Real phone camera simulation:
1. Sensor noise (2-8% std dev)
2. Rolling shutter effects
3. Hand shake motion blur
4. Phone AI processing (1.2-1.6x oversharpening)
5. Wide-angle distortion
6. Chromatic aberration
7. User filter adjustments
8. Social platform compression
9. Re-upload cycles
10. Accumulated artifacts
```

## Training Parameters

### **Optimized for Perceptual Fine-Tuning**
- **Lower Learning Rates**: Prevents overfitting to the fine-tuning dataset
- **Enhanced Perceptual Loss**: Focus on perceptual quality over pixel accuracy
- **Reduced GAN Loss**: Moderate adversarial training for fine-tuning
- **Small Batch Size**: 4 images per GPU with gradient accumulation
- **High EMA Decay**: 0.9999 for stable fine-tuning

### **Validation & Monitoring**
- **Frequent Validation**: Every 500 iterations during fine-tuning
- **Multiple Metrics**: PSNR, SSIM, and perceptual metrics
- **Regular Checkpointing**: Monitor progress and enable rollback

## Why This Approach Works

### **1. Realistic Degradation Training**
Instead of random degradations, the model learns on **realistic workflow chains** that represent actual image processing pipelines.

### **2. Identity Learning**
The 10% clean pass-through ensures the model **doesn't over-process** high-quality input.

### **3. Complete Coverage**
The workflow distribution covers the **entire spectrum** of real-world image degradation scenarios.

### **4. Perceptual Focus**
Loss weights are optimized for **perceptual quality** rather than pixel-perfect reconstruction.

## Expected Results

After training with this configuration:
- ✅ **Identity Preservation**: Model maintains quality of clean input
- ✅ **Realistic Restoration**: Handles actual workflow degradations
- ✅ **Perceptual Quality**: Enhanced visual quality for human observers
- ✅ **Generalization**: Works across different camera types and platforms

## Customization

You can adjust workflow probabilities based on your specific needs:
```yaml
predefined_sequences:
  phone_to_social:
    probability: 0.30  # Increase if you have more phone/social media data
  professional_camera:
    probability: 0.10  # Decrease if you have less professional data
```

This configuration provides a **scientifically sound foundation** for perceptual fine-tuning with realistic training data distribution.
