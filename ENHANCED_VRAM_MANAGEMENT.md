# Enhanced Dynamic VRAM Management System

## Overview

The Enhanced Dynamic VRAM Management System is a significant upgrade to the training automations that optimizes both **batch_size** and **lq_size** parameters dynamically during training. This system uses real-time VRAM measurements to maximize training efficiency while preventing out-of-memory errors.

## Key Features

### 🎯 Intelligent Priority System
- **Priority 1**: Increase `lq_size` first (better final metrics)
- **Priority 2**: Then increase `batch_size` (better training stability)
- **Safety Priority**: Decrease `batch_size` first when over target (less impact on metrics)

### 📊 Real-Time Optimization
- **Peak VRAM Tracking**: Monitors maximum VRAM usage during training
- **Dynamic Adjustments**: Adapts parameters based on actual VRAM measurements
- **Smart Cooldowns**: Prevents parameter thrashing with adjustment frequency limits

### 🚨 Enhanced OOM Recovery
- **Dual Parameter Recovery**: Adjusts both batch_size and lq_size after OOM
- **Aggressive Reduction**: Reduces parameters more conservatively after OOM events
- **Extended Cooldowns**: Longer adjustment periods after OOM recovery

### 📈 Comprehensive Statistics
- **VRAM History**: Tracks VRAM usage over time
- **Adjustment History**: Logs all parameter changes with reasons
- **Peak Usage Reporting**: Provides detailed VRAM utilization analytics

## Configuration

### Basic Configuration
```yaml
training_automations:
  enabled: true
  DynamicBatchSizeOptimizer:
    enabled: true
    target_vram_usage: 0.85          # Target 85% VRAM usage
    safety_margin: 0.05              # 5% safety buffer
    adjustment_frequency: 100        # Adjust every 100 iterations

    # Parameter bounds
    min_batch_size: 2                # Minimum batch size
    max_batch_size: 64               # Maximum batch size
    min_lq_size: 32                  # Minimum patch size (2x training)
    max_lq_size: 256                 # Maximum patch size

    vram_history_size: 50            # VRAM history tracking
```

### Advanced Configuration
```yaml
training_automations:
  enabled: true
  DynamicBatchSizeOptimizer:
    enabled: true
    target_vram_usage: 0.80          # Conservative 80% target
    safety_margin: 0.10              # Larger 10% safety buffer
    adjustment_frequency: 50         # More frequent adjustments

    # Tight parameter bounds for safety
    min_batch_size: 2
    max_batch_size: 32               # Conservative max
    min_lq_size: 64                  # Larger min patch size
    max_lq_size: 192                 # Conservative max

    vram_history_size: 100           # Extended history
    max_adjustments: 50              # Limit adjustments
```

## Parameter Bounds Guidelines

### For 2x Training
- **`lq_size` Range**: 32-256 (HR will be 2x this value)
- **`batch_size` Range**: 2-64 (RTX 3060 12GB optimized)

### Hardware-Specific Recommendations

#### RTX 3060 12GB (Recommended)
- Target VRAM: 80-85%
- lq_size bounds: 32-256
- batch_size bounds: 2-64

#### RTX 4090 24GB
- Target VRAM: 85-90%
- lq_size bounds: 64-512
- batch_size bounds: 4-128

#### RTX 3080 10GB
- Target VRAM: 75-80%
- lq_size bounds: 32-192
- batch_size bounds: 2-32

## Integration with Training Loop

The enhanced VRAM optimizer integrates seamlessly with the existing training infrastructure:

```python
# In training loop
if training_automations and "DynamicBatchSizeOptimizer" in training_automations.automations:
    vram_optimizer = training_automations.automations["DynamicBatchSizeOptimizer"]

    # Update VRAM monitoring
    batch_adjustment, lq_adjustment = vram_optimizer.update_vram_monitoring()

    if batch_adjustment is not None or lq_adjustment is not None:
        # Apply adjustments
        new_batch_size = max(vram_optimizer.min_batch_size,
                           min(vram_optimizer.max_batch_size,
                               vram_optimizer.current_batch_size + (batch_adjustment or 0)))
        new_lq_size = max(vram_optimizer.min_lq_size,
                        min(vram_optimizer.max_lq_size,
                            vram_optimizer.current_lq_size + (lq_adjustment or 0)))

        # Update dataloader and model parameters
        dataloader.update_batch_size(new_batch_size)
        dataloader.update_lq_size(new_lq_size)
```

## API Reference

### DynamicBatchSizeOptimizer Methods

#### `update_vram_monitoring() -> tuple[int | None, int | None]`
Returns suggested (batch_size_adjustment, lq_size_adjustment).

#### `set_current_parameters(batch_size: int, lq_size: int) -> None`
Set current parameters for monitoring.

#### `set_target_parameters(batch_size: int, lq_size: int) -> None`
Set target parameters for optimization.

#### `handle_oom_recovery(new_batch_size: int, new_lq_size: int) -> None`
Handle OOM recovery with dual parameter adjustment.

#### `get_vram_stats() -> dict[str, Any]`
Get comprehensive VRAM statistics including:
- `peak_usage`: Maximum VRAM usage observed
- `avg_usage`: Average VRAM usage
- `current_usage`: Current VRAM usage
- `current_batch_size`, `current_lq_size`: Current parameters
- `oom_recovery_count`: Number of OOM recoveries

## Benefits

### 🎯 Better Final Metrics
- Prioritizes `lq_size` increases for improved model performance
- Larger patches provide more contextual information
- Better feature extraction and representation learning

### ⚡ Improved Training Stability
- Dynamic `batch_size` optimization for stable training
- Adaptive adjustments prevent training disruptions
- OOM protection with intelligent recovery

### 📊 Data-Driven Optimization
- Uses actual VRAM measurements, not estimates
- Adapts to different training phases and model complexities
- Peak usage tracking for future optimization insights

### 🔧 Production Ready
- Comprehensive logging and statistics
- Safety bounds enforcement
- Graceful degradation and recovery mechanisms

## Testing

Run the test script to verify the enhanced VRAM management:

```bash
python test_enhanced_vram_management.py
```

This will demonstrate:
- Priority system behavior
- VRAM adjustment scenarios
- Peak tracking functionality
- OOM recovery mechanisms

## Migration from Old System

The enhanced system is backward compatible with existing configurations:

### Old Configuration
```yaml
training_automations:
  DynamicBatchSizeOptimizer:
    enabled: true
    target_vram_usage: 0.85
    min_batch_size: 1
    max_batch_size: 32
```

### New Configuration (Enhanced)
```yaml
training_automations:
  DynamicBatchSizeOptimizer:
    enabled: true
    target_vram_usage: 0.85
    min_batch_size: 2        # Safer minimum
    max_batch_size: 64       # Extended maximum
    min_lq_size: 32          # New parameter
    max_lq_size: 256         # New parameter
```

The system will work with existing configs, automatically applying sensible defaults for new parameters.

## Troubleshooting

### Common Issues

#### No Adjustments Happening
- Check `adjustment_frequency` (default: 100 iterations)
- Verify `enabled: true` in configuration
- Ensure `current_batch_size` and `current_lq_size` are set

#### OOM Still Occurring
- Reduce `target_vram_usage` to 0.75-0.80
- Increase `safety_margin` to 0.10
- Lower `max_batch_size` and `max_lq_size` bounds

#### Adjustments Too Frequent
- Increase `adjustment_frequency` to 200-500
- Increase `safety_margin` to reduce oscillation
- Check for other memory-intensive processes

### Performance Monitoring

Monitor VRAM optimization effectiveness:

```python
# Get VRAM statistics
vram_stats = vram_optimizer.get_vram_stats()
print(f"Peak VRAM: {vram_stats['peak_usage']:.1%}")
print(f"Adjustments: {vram_stats['adjustments']}")
print(f"OOM Recoveries: {vram_stats['oom_recovery_count']}")
```

## Future Enhancements

Planned improvements:
- **Multi-GPU Support**: Extend to distributed training setups
- **Learning-Based Optimization**: ML models to predict optimal adjustments
- **Architecture Detection**: Auto-detect model complexity for bounds
- **Training Phase Awareness**: Different optimization strategies per phase

---

This enhanced VRAM management system represents a significant advancement in automated training optimization, providing both performance improvements and operational stability for production training workflows.
