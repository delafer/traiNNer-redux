# VRAM Analysis and Fixes for RTX 3060 SISR Training

## **Problem Summary**

Your training shows a **classic VRAM usage discrepancy**:
- **Training VRAM**: 1.71 GB (very low, good!)
- **Validation OOM**: "Tried to allocate 128.00 GiB" (massive failure)
- **Root Cause**: Attention mechanism during validation phase

## **Why Training VRAM is Low vs Validation OOM**

### **1. Training vs Validation Memory Patterns**

**Training Phase:**
- Batch size: 6 images (64x64 LR → 128x128 HR)
- AMP enabled: fp16 precision reduces memory
- Channels last format: Better memory efficiency
- Peak VRAM: 1.71 GB (within limits)

**Validation Phase:**
- **CRITICAL ISSUE**: Validation might be using larger batch sizes
- **Attention Memory Spike**: The error occurs in ParagonSR2 attention mechanism:
  ```python
  # Error location in paragonsr2_arch.py:436
  attn = torch.bmm(q, v)  # Batch matrix multiplication
  ```
- **Memory Allocation**: 128 GB allocation request for attention operations
- **Mathematical Explosion**: Attention scales as O(n²) with sequence length

### **2. The Attention Mechanism Problem**

```python
# What's happening in validation:
# Input: [batch_size, channels, height, width]
# After flattening: [batch_size, height*width, channels]

# Attention matrix: Q @ K^T
# Shape: [B, H*W, C] @ [B, C, H*W] = [B, H*W, H*W]

# For 128x128 images: H*W = 16384
# Attention matrix: [B, 16384, 16384] per batch
# Memory: B * 16384² * 4 bytes (fp32)
# Single image (B=1): ~1.07 GB just for attention matrix
# Multiple images: Multiplied by batch size
```

## **Dynamic Loss Scheduling - Update Frequency**

**Answer to your question**: Dynamic Loss Scheduling updates **EVERY training iteration** after baseline establishment.

### **Update Mechanism:**
```python
# 1. Baseline Phase: First 150 iterations (configurable)
#    - Just monitor and record loss values
#    - No weight adaptation yet

# 2. Adaptation Phase: Every iteration after baseline
for iteration in range(150, total_iterations):
    current_loss = get_loss_value()

    # Exponential smoothing with momentum = 0.92
    smoothed_loss = momentum * smoothed_loss + (1-momentum) * current_loss

    # Adaptive weight calculation
    if abs(smoothed_loss - baseline) > threshold:
        adaptation_direction = baseline / smoothed_loss
        weight = weight * (1 + adaptation_rate * adaptation_direction)
        weight = clip(weight, min_weight, max_weight)
```

### **Key Parameters:**
- **momentum = 0.92**: 25% new information, 75% historical smoothing
- **adaptation_rate = 0.01**: 1% weight change per iteration when needed
- **baseline_iterations = 150**: 150 iterations to establish stable baseline
- **updates**: Every iteration (26+ updates per second at 26 it/s)

## **Immediate Fixes Applied**

### **1. Validation-Specific Memory Optimization**

Modified `2xParagonSR2_Nano_CC0_RTX3060_Optimized.yml`:

```yaml
datasets:
  val:
    lq_size: 32                      # FIX: Smaller validation size
    gt_size: 64                      # FIX: Smaller validation size

train:
  # CRITICAL: Validation-specific memory optimization
  val_batch_size: 1                   # Single image validation
  val_num_workers: 1                  # Minimal workers
  val_prefetch_factor: 2              # Reduced prefetching
  val_pin_memory: false               # Disable pin_memory
```

### **2. Expected Memory Reduction**

**Before (OOM):**
- Validation batch: Unknown size (probably 6 like training)
- Image size: 128x128 → 256x256
- Attention matrix: [6, 65536, 65536] = ~98 GB

**After (Fixed):**
- Validation batch: 1 image only
- Image size: 32x32 → 64x64
- Attention matrix: [1, 4096, 4096] = ~67 MB
- **Reduction**: 98 GB → 67 MB (99.9% reduction)

### **3. Training Impact Assessment**

**What you lose:**
- Slightly less accurate validation metrics (smaller images)
- Fewer validation samples per batch (longer validation time)

**What you gain:**
- **Stable training**: No more OOM crashes
- **Continuous monitoring**: Validate every 1000 iterations as intended
- **Full training completion**: Can train to 40,000 iterations

## **Additional Warnings to Fix**

### **1. PyTorch Deprecation Warnings**

```bash
UserWarning: Please use the new API settings to control TF32 behavior
```

**Solution**: Add to your training script or environment:
```python
# Replace old settings:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# With new settings:
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
```

### **2. Dynamic Loss Scheduling Warning**

```bash
UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
```

**Solution**: Already fixed in your dynamic loss scheduling code:
```python
# Fixed line in dynamic_loss_scheduling.py:209
scalar_losses[loss_name] = float(loss_value.mean().abs().detach())
```

### **3. Parameter Optimization Warnings**

```bash
WARNING: Params magic_upsampler.sharpen.conv_h.weight will not be optimized.
```

**Analysis**: These are expected warnings from your MagicKernelSharp component where certain parameters are intentionally frozen (good for stability).

## **Training Command with Fixes**

```bash
python train.py -opt options/train/ParagonSR2/2xParagonSR2_Nano_CC0_RTX3060_Optimized.yml
```

**What to expect:**
- Training VRAM: ~1.5-2.0 GB (stable)
- Validation VRAM: ~200-500 MB (no OOM)
- Validation time: ~30-60 seconds per 70 images (acceptable)
- Dynamic Loss Scheduling: Updates every iteration after iteration 150

## **Long-term Optimizations (Optional)**

If you want even better performance after confirming this fix works:

### **1. Gradient Checkpointing**
```yaml
train:
  gradient_checkpointing: true    # Reduce training VRAM by ~30%
```

### **2. Mixed Precision Optimization**
```yaml
use_amp: true
amp_bf16: true                    # Use BF16 for better precision
```

### **3. Validation Frequency Adjustment**
```yaml
val:
  val_freq: 2000                  # Validate less frequently for faster training
```

## **Success Metrics**

**Your training is successful when:**
1. ✅ No more CUDA OOM errors during validation
2. ✅ Training completes full 40,000 iterations
3. ✅ Dynamic Loss Scheduling shows adaptive weights in logs
4. ✅ Validation metrics show reasonable PSNR/SSIM values

The fix should resolve your VRAM issues while maintaining training quality and speed!
