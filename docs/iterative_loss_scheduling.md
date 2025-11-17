# Iterative Loss Scheduling Framework Extension

## Overview

The traiNNer-redux framework has been extended to support **iteration-based loss scheduling**, enabling sophisticated training strategies with dynamic loss weight changes throughout training.

## Features

### New Loss Scheduling Parameters

- **`start_iter`**: When to begin applying this loss (default: 0)
- **`target_iter`**: When to reach target_weight (default: start_iter)
- **`target_weight`**: Final weight to ramp to (default: loss_weight)
- **`disable_after`**: When to completely disable the loss (default: None)
- **`schedule_type`**: Type of scheduling ('linear', 'cosine', 'step') (default: 'linear')

### Supported Schedule Types

1. **Linear**: Smooth linear interpolation from base weight to target weight
2. **Cosine**: Cosine easing for gentler transitions
3. **Step**: Sharp transition to target weight at target_iter

## Usage Example

```yaml
# In your training config:
losses:
  # Phase 1: Start with lower perceptual weight
  - type: convnextperceptualloss
    loss_weight: 0.12
    start_iter: 0
    target_iter: 20000
    target_weight: 0.26
    schedule_type: "linear"

  # Phase 2: Activate GAN after 20k iterations
  - type: ganloss
    gan_type: r3gan
    loss_weight: 0.00
    start_iter: 20000
    target_weight: 0.08
    schedule_type: "step"

  # Phase 3: Remove artifact suppression after 50k
  - type: ldlloss
    loss_weight: 0.5
    disable_after: 50000
```

## Implementation Details

### Core Components

1. **`IterativeLossWrapper`** (`traiNNer/losses/iterative_loss_wrapper.py`)
   - Wraps any loss function with iteration-aware weight scheduling
   - Automatically detects scheduling parameters in loss configurations
   - Provides smooth weight transitions for training stability

2. **Enhanced Loss Builder** (`traiNNer/losses/__init__.py`)
   - Detects scheduling parameters in loss configurations
   - Automatically wraps applicable losses with `IterativeLossWrapper`
   - Maintains backward compatibility with existing configs

3. **Training Loop Integration** (`traiNNer/models/sr_model.py`)
   - Passes `current_iter` to wrapped losses during training
   - Handles special cases for GAN losses and R3GAN
   - Maintains compatibility with all existing loss types

### How It Works

1. **Detection**: When `build_loss()` is called, it checks for scheduling parameters
2. **Wrapping**: If scheduling parameters are found, the loss is wrapped with `IterativeLossWrapper`
3. **Training**: During training, the wrapper calculates the effective weight for the current iteration
4. **Application**: The effective weight is applied to the loss value before backpropagation

## Advanced Features

### Weight Calculation Logic

The framework handles several scenarios:

- **Before start_iter**: Weight is 0.0 (loss disabled)
- **During ramp**: Weight interpolates based on schedule type
- **After target_iter**: Weight remains at target_weight
- **After disable_after**: Weight is 0.0 (loss permanently disabled)

### Schedule Types Comparison

| Schedule Type | Description | Use Case |
|---------------|-------------|----------|
| `linear` | Linear interpolation | Gradual weight changes |
| `cosine` | Cosine easing | Smooth, gentle transitions |
| `step` | Sharp transition | Binary activation/deactivation |

### Compatibility

- ✅ All existing loss types are supported
- ✅ Backward compatible with static loss configurations
- ✅ Works with GAN losses (including R3GAN)
- ✅ Compatible with multi-component losses
- ✅ Handles both scalar and dict-returning losses

## Benefits

1. **Sophisticated Training Strategies**: Implement phase-based training approaches
2. **Dynamic Loss Control**: Adjust loss importance based on training progress
3. **Improved Stability**: Smooth weight transitions prevent training instability
4. **Better Visual Results**: Enable controlled evolution from fidelity to perceptual quality
5. **Easy Configuration**: Simple YAML-based scheduling parameters

## Example Training Strategy

```yaml
# Training Strategy for Maximum Visual Distinction
Phase 1 (0-20k):    # Structure learning
  - High pixel loss weight
  - Moderate perceptual loss (ramping up)
  - No GAN loss

Phase 2 (20k-50k):  # Perceptual enhancement
  - Reduced pixel loss
  - Increased perceptual loss (ramped up)
  - GAN loss activation (ramping up)

Phase 3 (50k+):     # Fine-tuning
  - Minimal artifact suppression
  - Fine-tuned perceptual balance
  - Full GAN training
```

This approach creates visually distinct models that evolve from fidelity-focused to perceptually enhanced, exactly as intended in your advanced training configuration.

## Technical Notes

- The wrapper transparently handles the underlying loss computation
- Performance overhead is minimal (only weight calculation)
- All PyTorch operations remain unchanged
- Memory usage is not increased significantly

## Migration Guide

### Existing Configs
No changes needed! Configs without scheduling parameters work exactly as before.

### New Configs
Add scheduling parameters to any loss that needs dynamic weight control:

```yaml
# Before (static weight)
- type: perceptual_loss
  loss_weight: 0.1

# After (dynamic weight)
- type: perceptual_loss
  loss_weight: 0.1
  start_iter: 5000
  target_iter: 15000
  target_weight: 0.2
```

The framework extension is ready for immediate use with your sophisticated training configurations!
