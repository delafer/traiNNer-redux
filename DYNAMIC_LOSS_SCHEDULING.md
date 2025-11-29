# Dynamic Loss Scheduling System

## Overview

The Dynamic Loss Scheduling system is a sophisticated training enhancement that automatically adjusts loss weights based on current loss values during training. This prevents loss dominance, maintains training balance, and adapts to changing dynamics automatically.

## Key Benefits

- **Automatic Loss Balance**: Prevents one loss from overwhelming others
- **Training Stability**: Reduces risk of training instabilities from unbalanced gradients
- **Better Convergence**: Maintains optimal training dynamics throughout the process
- **Reduced Manual Tuning**: Automatically adapts to dataset and model characteristics
- **GAN Stability**: Particularly beneficial for GAN training where discriminator/generator balance is critical

## How It Works

The system monitors loss magnitudes using exponential smoothing and adjusts weights to:

1. **Reduce weights** for losses that are increasing rapidly (preventing dominance)
2. **Increase weights** for losses that are decreasing (encouraging progress)
3. **Maintain stability** through safety bounds and momentum-based tracking
4. **Establish baselines** over initial iterations before making adjustments

## Configuration

Add the `dynamic_loss_scheduling` section to your training configuration:

```yaml
train:
  dynamic_loss_scheduling:
    enabled: true
    momentum: 0.9
    adaptation_rate: 0.01
    min_weight: 1e-6
    max_weight: 100.0
    adaptation_threshold: 0.1
    baseline_iterations: 100
    enable_monitoring: true
```

### Parameter Details

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `enabled` | Enable/disable dynamic loss scheduling | `true` | Boolean |
| `momentum` | Exponential smoothing factor for loss tracking | `0.9` | 0.0-1.0 |
| `adaptation_rate` | Rate of weight adaptation per iteration | `0.01` | 0.001-0.1 |
| `min_weight` | Minimum possible weight multiplier | `1e-6` | >0 |
| `max_weight` | Maximum possible weight multiplier | `100.0` | >min_weight |
| `adaptation_threshold` | Minimum relative loss change to trigger adaptation | `0.1` | 0.01-1.0 |
| `baseline_iterations` | Iterations to establish baseline before adapting | `100` | 50-1000 |
| `enable_monitoring` | Enable detailed logging and monitoring | `true` | Boolean |

## Usage Examples

### 1. Basic Fidelity Training

```yaml
train:
  dynamic_loss_scheduling:
    enabled: true
    momentum: 0.9
    adaptation_rate: 0.01
    min_weight: 1e-6
    max_weight: 10.0
    baseline_iterations: 100

  losses:
    - type: l1loss
      loss_weight: 1.0
    - type: ssimloss
      loss_weight: 0.05
    - type: perceptualloss
      loss_weight: 0.5
```

### 2. GAN Training with Enhanced Stability

```yaml
train:
  dynamic_loss_scheduling:
    enabled: true
    momentum: 0.9
    adaptation_rate: 0.01      # Conservative for GAN stability
    min_weight: 1e-6
    max_weight: 10.0           # Conservative maximum for GANs
    adaptation_threshold: 0.15  # Higher threshold for stability
    baseline_iterations: 200    # Longer baseline for GANs
    enable_monitoring: true

  losses:
    - type: l1loss
      loss_weight: 1.0
    - type: perceptualloss
      loss_weight: 0.5
    - type: ganloss
      gan_type: r3gan
      loss_weight: 0.05
    - type: featurematchingloss
      loss_weight: 0.1
```

### 3. Aggressive Adaptation for Small Models

```yaml
train:
  dynamic_loss_scheduling:
    enabled: true
    momentum: 0.8               # Faster adaptation for small models
    adaptation_rate: 0.02       # Faster adaptation
    min_weight: 1e-6
    max_weight: 50.0
    adaptation_threshold: 0.05   # More responsive
    baseline_iterations: 50      # Quicker baseline establishment

  losses:
    - type: l1loss
      loss_weight: 1.0
    - type: ssimloss
      loss_weight: 0.05
```

## Monitoring and Debugging

When `enable_monitoring: true`, the system provides:

### 1. Console Logging
- Initial baseline establishment
- Significant weight changes
- Adaptation statistics

### 2. TensorBoard Metrics
The system automatically logs dynamic weight values as:
- `dynamic_weight_l1_loss`
- `dynamic_weight_ssim_loss`
- `dynamic_weight_perceptual_loss`
- etc.

### 3. Training State
Monitor these indicators:
- **Adaptation Count**: Number of significant weight adjustments
- **Weight Stability**: Recent weight changes should decrease over time
- **Loss Balance**: Individual losses should maintain reasonable proportions

## Best Practices

### When to Use
- **Complex Loss Combinations**: When using multiple loss functions
- **GAN Training**: To prevent discriminator overpowering
- **Unstable Training**: When manually tuned weights lead to instability
- **Different Datasets**: Automatic adaptation to dataset characteristics

### When to Avoid
- **Simple Single-Loss Training**: No benefit for single loss configurations
- **Extremely Short Training**: Insufficient time for adaptation
- **Critical Weight Ratios**: When specific loss ratios are scientifically required

### Tuning Guidelines

**For Stability (GANs, Complex Training)**:
```yaml
dynamic_loss_scheduling:
  enabled: true
  momentum: 0.9
  adaptation_rate: 0.01
  max_weight: 10.0
  adaptation_threshold: 0.15
  baseline_iterations: 200
```

**For Responsiveness (Small Models, Quick Experiments)**:
```yaml
dynamic_loss_scheduling:
  enabled: true
  momentum: 0.8
  adaptation_rate: 0.02
  max_weight: 50.0
  adaptation_threshold: 0.05
  baseline_iterations: 50
```

**For Conservative Adaptation (Large Models, Long Training)**:
```yaml
dynamic_loss_scheduling:
  enabled: true
  momentum: 0.95
  adaptation_rate: 0.005
  max_weight: 20.0
  adaptation_threshold: 0.2
  baseline_iterations: 300
```

## Integration with Existing Features

### Iterative Loss Wrapper
Dynamic loss scheduling works seamlessly with existing iteration-based loss scheduling:

```yaml
losses:
  - type: ganloss
    gan_type: r3gan
    loss_weight: 0.05
    start_iter: 10000          # Iterative scheduling
    target_iter: 15000
    # Dynamic scheduling will still operate after iterative scheduling
```

### Gradient Clipping
Compatible with gradient clipping for maximum training stability:

```yaml
train:
  grad_clip: true
  dynamic_loss_scheduling:
    enabled: true
    # ... configuration
```

### Mixed Precision
Fully compatible with AMP and BF16 training:

```yaml
use_amp: true
amp_bf16: true
train:
  dynamic_loss_scheduling:
    enabled: true
    # ... configuration
```

## Technical Details

### Algorithm
1. **Baseline Establishment**: Monitor loss values for N iterations without adaptation
2. **Loss Tracking**: Use exponential smoothing to track loss dynamics
3. **Velocity Calculation**: Compute rate of change for each loss
4. **Weight Adjustment**: Apply proportional adjustments based on loss behavior
5. **Bounds Enforcement**: Ensure weights stay within safe limits

### Mathematical Formulation
- **Smoothing**: `smoothed_loss = α * current_loss + (1 - α) * prev_smoothed`
- **Velocity**: `velocity = momentum * prev_velocity + (1 - momentum) * loss_change`
- **Adjustment**: `weight = base_weight * (1 + adaptation_rate * adjustment_factor)`

### Performance Impact
- **Computational Overhead**: < 1% of total training time
- **Memory Usage**: Minimal additional GPU memory
- **Convergence**: Often improves final convergence quality

## Troubleshooting

### Common Issues

**Weights oscillating rapidly**:
- Increase `adaptation_threshold`
- Decrease `adaptation_rate`
- Increase `baseline_iterations`

**No adaptation happening**:
- Check `enabled: true`
- Ensure losses have sufficient magnitude
- Verify `adaptation_threshold` isn't too high

**Loss dominance persists**:
- Decrease `max_weight` limit
- Increase `adaptation_rate`
- Check `adaptation_threshold`

### Debug Commands

Monitor scheduler state:
```python
# In training loop or debugger
scheduler_stats = model.dynamic_loss_scheduler.get_monitoring_stats()
print(f"Current weights: {scheduler_stats['current_weights']}")
print(f"Adaptation count: {scheduler_stats['adaptation_count']}")
```

Reset scheduler if needed:
```python
model.dynamic_loss_scheduler.reset(keep_baseline=False)
```

## Examples and Templates

### Reference Configurations
1. **`2xParagonSR2_Nano_Fidelity_DynamicLoss.yml`**: Clean fidelity training with dynamic loss scheduling
2. **`2xParagonSR2_Nano_Perceptual_DynamicGAN.yml`**: GAN training with enhanced stability

### Migration from Static Weights
Simply add the dynamic loss scheduling configuration to your existing training config. The system will automatically adapt your static weights.

## Conclusion

Dynamic Loss Scheduling provides automatic, intelligent weight adaptation that improves training stability and final results. It's particularly valuable for:

- GAN training stability
- Complex multi-loss configurations
- Training across different datasets
- Reducing manual hyperparameter tuning

The system is designed to be safe, predictable, and compatible with all existing traiNNer features.
