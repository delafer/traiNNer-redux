# Dataset Comparison Testing Guide

## Executive Summary

**Yes, you should disable automatic loss scheduling and parameter setting for fair dataset comparison.**

## What to Disable

### 1. Dynamic Loss Scheduling with Auto-Calibration
**Current (unfair):**
```yaml
dynamic_loss_scheduling:
  enabled: true
  auto_calibrate: true  # ❌ Detects dataset complexity per dataset
```

**Fixed (fair):**
```yaml
dynamic_loss_scheduling:
  enabled: false  # ✅ Use fixed loss weights
```

### 2. Training Automations
**Current (unfair):**
```yaml
training_automations:
  intelligent_learning_rate_scheduler:
    enabled: true  # ❌ Adapts LR per dataset
  dynamic_batch_size_optimizer:
    enabled: true  # ❌ Changes batch size per dataset
```

**Fixed (fair):**
```yaml
# Remove entire training_automations section
```

## What to Keep (Essential for Stability)

- **EMA**: `ema_decay: 0.999` (model improvement)
- **Gradient Clipping**: `grad_clip: true` (stability)
- **AMP**: Mixed precision training (performance)
- **Fixed LR Schedule**: MultiStepLR with fixed milestones
- **Fixed Optimizer**: AdamW with fixed parameters

## Why Disable Auto-Calibration?

The system currently detects dataset complexity and adjusts parameters differently:
- **CC0**: Gets one set of optimized parameters
- **BHI**: Gets different parameters based on its complexity
- **PSISRD**: Gets yet another parameter set

This means you're not comparing datasets - you're comparing dataset + its optimized training setup.

## Recommended Base Configuration

```yaml
name: Dataset_Benchmark_Base
train:
  # Core stability features
  ema_decay: 0.999
  grad_clip: true

  # Fixed optimization
  optim_g:
    type: AdamW
    lr: 2e-4  # Fixed learning rate
    weight_decay: 1e-4
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [30000, 35000]  # Fixed schedule
    gamma: 0.5

  total_iter: 30000

  # DISABLED for fair comparison
  dynamic_loss_scheduling:
    enabled: false

  losses:
    - type: l1loss
      loss_weight: 1.0  # Fixed weight

val:
  val_freq: 1000  # Frequent validation for good graphs
  metrics_enabled: true
```

## Testing Protocol

1. **Create identical configs** for all datasets (only change dataset paths)
2. **Run 30,000 iterations** with validation every 1,000 iterations
3. **Compare validation curves** for:
   - PSNR convergence speed and final scores
   - SSIM convergence speed and final scores
   - Training stability

## Expected Results

With disabled auto-calibration, you'll see:
- **True dataset quality differences** in validation metrics
- **Fair comparison** of convergence speeds
- **Authentic performance rankings** of your filtered datasets

## Quick Fix for Your Current Configs

Simply add `enabled: false` to your dynamic loss scheduling sections:

```yaml
# In all three config files
dynamic_loss_scheduling:
  enabled: false  # Add this line
  auto_calibrate: false  # Change to false
```

This single change will make your dataset comparison meaningful and fair.
