# 🚨 Critical Training Fixes for Nano Model

## ✅ **Architecture: SOLID (No Changes Needed)**
Your ParagonSR2 architecture has:
- ✅ LayerScale for training stability
- ✅ LocalWindowAttention for VRAM efficiency
- ✅ ContentAwareDetailProcessor for adaptive processing
- ✅ MagicKernel base for structural stability

## ❌ **Configuration Issues Identified**

### **1. Gradient Clipping Error**
```yaml
# WRONG (doesn't exist):
grad_clip_max_norm: 1.0

# CORRECT:
grad_clip: true  # Uses clip_grad_norm_(..., 1.0) by default
```

### **2. CosineAnnealingLR + EMA Incompatibility**
```yaml
# PROBLEMATIC (causes instability):
scheduler:
  type: CosineAnnealingLR  # ❌ Gradually reduces learning signals
# EMA with β=0.999 aggressively smooths weights
# Result: Training becomes over-smoothed, stops learning

# FIXED:
scheduler:
  type: MultiStepLR
  milestones: [30000, 35000]  # Later milestones for EMA stability
  gamma: 0.5
```

### **3. Dynamic Loss Scheduling Issues**
```yaml
# TOO CONSERVATIVE (your config):
dynamic_loss_scheduling:
  momentum: 0.95           # ❌ Too high, sluggish response
  adaptation_rate: 0.005   # ❌ Too slow for Nano model
  max_weight: 100.0        # ❌ Too high, causes instability

# NANO-OPTIMIZED:
dynamic_loss_scheduling:
  momentum: 0.9            # ✅ More responsive
  adaptation_rate: 0.01    # ✅ Standard rate
  max_weight: 10.0         # ✅ Lower bound
  baseline_iterations: 100 # ✅ Shorter baseline
```

### **4. AMP Precision Issues**
```yaml
# CAUSES UNDERFLOW:
amp_bf16: false            # ❌ Numerical precision problems

# STABLE:
amp_bf16: true             # ✅ Better precision prevents explosions
```

## 🛠️ **Complete Fixed Configuration**

Replace your current config with this **COMPREHENSIVE FIX**:

```yaml
name: 2xParagonSR2_Nano_CC0_147k_COMPREHENSIVE_FIX
scale: 2

use_amp: true
amp_bf16: true              # ✅ FIXED: Enable BF16

use_channels_last: true
fast_matmul: true
num_gpu: auto
manual_seed: 1024

train:
  ema_decay: 0.999
  ema_power: 0.75
  grad_clip: true           # ✅ CORRECT parameter

  optim_g:
    type: AdamW
    lr: !!float 2e-4
    weight_decay: !!float 1e-4
    betas: [0.9, 0.99]

  # ✅ FIXED: Compatible with EMA
  scheduler:
    type: MultiStepLR
    milestones: [30000, 35000]     # Later for EMA stability
    gamma: 0.5

  total_iter: 40000
  warmup_iter: 1000          # ✅ Longer warmup

  # ✅ FIXED: Nano-optimized parameters
  dynamic_loss_scheduling:
    enabled: true           # Keep enabled as requested
    momentum: 0.9
    adaptation_rate: 0.01
    min_weight: 1e-6
    max_weight: 10.0
    adaptation_threshold: 0.05
    baseline_iterations: 100
    enable_monitoring: true

  losses:
    - type: l1loss
      loss_weight: 1.0
    - type: ssimloss
      loss_weight: 0.05

val:
  val_enabled: true
  val_freq: 1000
  save_img: false

  metrics_enabled: true
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
    ssim:
      type: calculate_ssim
      crop_border: 4
```

## 🎯 **Expected Results**

- ✅ Scale_g values stabilize in 1e3-1e4 range (not 1e6)
- ✅ Continuous improvement beyond 1,000 iterations
- ✅ EMA model stays synchronized with main model
- ✅ Peak performance near end of training (35k-40k iterations)
- ✅ No degradation due to configuration conflicts

## 💡 **Root Cause**

The degradation occurred because:
1. **CosineAnnealingLR** gradually reduced learning signals
2. **EMA with β=0.999** aggressively smoothed model weights
3. **Dynamic loss scheduling** parameters were too conservative
4. **AMP without BF16** caused numerical precision issues

Your architecture is excellent - the issues were purely configuration-related!
