# Commented Configuration Guide

## 🎯 **Philosophy: Simple by Default, Powerful when Needed**

The training configurations follow a **minimal-first, transparent** approach where:
- **Active configuration**: Only essential `enabled: true` flags
- **Documentation**: All available parameters shown as comments
- **Flexibility**: Users can uncomment and customize any parameter

## 📋 **Configuration Structure**

### **Current Active Configuration (Minimal)**
```yaml
dynamic_loss_scheduling:
  enabled: true

training_automations:
  intelligent_learning_rate_scheduler:
    enabled: true

  dynamic_batch_size_optimizer:
    enabled: true

  early_stopping:
    enabled: true

  adaptive_gradient_clipping:
    enabled: true
```

### **Full Available Options (Commented)**
```yaml
dynamic_loss_scheduling:
  enabled: true
  # momentum: 0.9                    # Smooth adaptation (0.0-1.0)
  # adaptation_rate: 0.01            # Adaptive rate per iteration
  # min_weight: 1e-6                 # Minimum possible weight
  # max_weight: 100.0                # Maximum possible weight
  # adaptation_threshold: 0.05       # More sensitive adaptation
  # baseline_iterations: 200         # Establish baseline before adapting
  # enable_monitoring: true          # Detailed logging
  # auto_calibrate: true             # Auto-detect dataset complexity

training_automations:
  intelligent_learning_rate_scheduler:
    enabled: true
    # strategy: "adaptive"            # adaptive, cosine, exponential, plateau
    # adaptation_frequency: 1000      # Check every 1000 iterations
    # improvement_threshold: 0.001    # Minimum improvement to continue current LR
    # patience: 2000                  # LR scheduling patience
    # max_lr_factor: 2.0              # Maximum LR multiplier
    # min_lr_factor: 0.1              # Minimum LR multiplier

  dynamic_batch_size_optimizer:
    enabled: true
    # target_vram_usage: 0.85         # Use 85% of available VRAM
    # safety_margin: 0.05            # Keep 5% free for stability
    # adjustment_frequency: 500       # Check every 500 iterations
    # min_batch_size: 1              # Minimum batch size
    # max_batch_size: 32             # Maximum batch size

  early_stopping:
    enabled: true
    # patience: 3000                  # Wait 3000 iterations without improvement
    # min_improvement: 0.0005         # Minimum PSNR improvement threshold
    # metric: "val/psnr"              # Monitor validation PSNR
    # save_best: true                 # Save best performing model
    # min_iterations: 5000            # Minimum iterations before stopping
    # warmup_iterations: 1000         # Warmup period

  adaptive_gradient_clipping:
    enabled: true
    # initial_threshold: 1.0          # Auto-calibrated based on architecture
    # min_threshold: 0.1              # Auto-calibrated based on architecture
    # max_threshold: 10.0             # Auto-calibrated based on architecture
    # adjustment_factor: 1.2          # Auto-calibrated based on architecture
    # monitoring_frequency: 50        # Auto-calibrated based on architecture
    # gradient_history_size: 100      # Auto-calibrated based on architecture
```

## 🛠️ **How to Customize**

### **1. Uncomment Parameters**
To customize any parameter, simply remove the `#` and space:

```yaml
# Before:
# target_vram_usage: 0.85         # Use 85% of available VRAM

# After:
target_vram_usage: 0.90         # Use 90% of available VRAM
```

### **2. Common Customizations**

#### **Increase VRAM Usage**
```yaml
dynamic_batch_size_optimizer:
  enabled: true
  target_vram_usage: 0.90         # Use 90% instead of 85%
  safety_margin: 0.03            # Keep only 3% free
```

#### **Adjust Early Stopping**
```yaml
early_stopping:
  enabled: true
  patience: 5000                  # Wait longer before stopping
  min_improvement: 0.001         # Require larger improvements
  metric: "val/ssim"              # Monitor SSIM instead of PSNR
```

#### **Customize Gradient Clipping**
```yaml
adaptive_gradient_clipping:
  enabled: true
  initial_threshold: 0.8          # Start with lower threshold
  max_threshold: 5.0              # Lower maximum threshold
  monitoring_frequency: 25        # Check more frequently
```

#### **Adjust Loss Scheduling**
```yaml
dynamic_loss_scheduling:
  enabled: true
  adaptation_rate: 0.005          # Slower adaptation
  adaptation_threshold: 0.02      # Less sensitive changes
  baseline_iterations: 500        # Longer baseline period
```

## 📚 **Parameter Reference**

### **Dynamic Loss Scheduling**
| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `momentum` | 0.9 | Smoothing factor for adaptations | 0.8-0.95 |
| `adaptation_rate` | 0.01 | Rate of weight changes per iteration | 0.001-0.05 |
| `min_weight` | 1e-6 | Minimum loss weight | 1e-8 to 1e-4 |
| `max_weight` | 100.0 | Maximum loss weight | 10 to 1000 |
| `adaptation_threshold` | 0.05 | Sensitivity to loss changes | 0.01-0.1 |
| `baseline_iterations` | 200 | Iterations to establish baseline | 100-1000 |

### **Learning Rate Scheduler**
| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `strategy` | "adaptive" | LR adjustment strategy | "adaptive", "cosine", "exponential" |
| `adaptation_frequency` | 1000 | Check interval in iterations | 500-2000 |
| `improvement_threshold` | 0.001 | Minimum improvement for LR maintenance | 0.0001-0.01 |
| `patience` | 2000 | LR reduction patience | 1000-5000 |
| `max_lr_factor` | 2.0 | Maximum LR multiplier | 1.5-5.0 |
| `min_lr_factor` | 0.1 | Minimum LR multiplier | 0.01-0.5 |

### **Batch Size Optimizer**
| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `target_vram_usage` | 0.85 | Target VRAM utilization | 0.7-0.95 |
| `safety_margin` | 0.05 | VRAM safety buffer | 0.02-0.1 |
| `adjustment_frequency` | 500 | Check interval in iterations | 100-1000 |
| `min_batch_size` | 1 | Minimum batch size | 1-8 |
| `max_batch_size` | 32 | Maximum batch size | 16-128 |

### **Early Stopping**
| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `patience` | 3000 | Iterations to wait for improvement | 1000-10000 |
| `min_improvement` | 0.0005 | Minimum metric improvement | 0.0001-0.01 |
| `metric` | "val/psnr" | Metric to monitor | "val/psnr", "val/ssim" |
| `min_iterations` | 5000 | Minimum iterations before stopping | 1000-20000 |
| `warmup_iterations` | 1000 | Warmup period before stopping checks | 500-5000 |

### **Adaptive Gradient Clipping**
| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `initial_threshold` | 1.0 | Starting gradient norm threshold | 0.1-10.0 |
| `min_threshold` | 0.1 | Minimum allowed threshold | 0.01-1.0 |
| `max_threshold` | 10.0 | Maximum allowed threshold | 1.0-100.0 |
| `adjustment_factor` | 1.2 | Threshold adjustment factor | 1.1-2.0 |
| `monitoring_frequency` | 50 | Check interval in iterations | 10-200 |
| `gradient_history_size` | 100 | History for pattern analysis | 50-500 |

## 🎯 **Architecture-Specific Defaults**

### **Nano Models**
- **Gradient Threshold**: 1.0 (higher - more aggressive)
- **VRAM Usage**: 85% (more aggressive)
- **Early Stopping Patience**: 3000 (shorter)
- **Batch Size Range**: 1-32 (wider range)

### **S Models**
- **Gradient Threshold**: 0.8 (lower - more conservative)
- **VRAM Usage**: 80% (more conservative)
- **Early Stopping Patience**: 5000 (longer)
- **Batch Size Range**: 1-16 (smaller range)

## 🚀 **Quick Start Examples**

### **Maximum Performance (Conservative)**
```yaml
dynamic_batch_size_optimizer:
  enabled: true
  target_vram_usage: 0.75         # Conservative VRAM usage
  safety_margin: 0.10            # Larger safety buffer

early_stopping:
  enabled: true
  patience: 10000                # Very patient
  min_improvement: 0.0001        # Very sensitive to improvements
```

### **Fast Training (Aggressive)**
```yaml
early_stopping:
  enabled: true
  patience: 1000                 # Quick stopping
  min_improvement: 0.005        # Less sensitive

adaptive_gradient_clipping:
  enabled: true
  initial_threshold: 2.0        # Higher threshold
  monitoring_frequency: 25      # More frequent monitoring
```

### **Memory-Constrained Environment**
```yaml
dynamic_batch_size_optimizer:
  enabled: true
  target_vram_usage: 0.70        # Conservative VRAM usage
  safety_margin: 0.15           # Large safety buffer
  min_batch_size: 1
  max_batch_size: 8             # Limit maximum batch size
```

## 💡 **Tips for Customization**

1. **Start Simple**: Begin with defaults and adjust one parameter at a time
2. **Monitor Training**: Check logs for automation decisions before making changes
3. **Test Changes**: Validate changes on a shorter training run before full training
4. **Document Customizations**: Note why you made specific parameter changes
5. **Architecture Awareness**: S models typically need more conservative settings

## 🎉 **The Best of Both Worlds**

This approach gives you:
- ✅ **Simplicity**: Minimal configuration for most users
- ✅ **Transparency**: All options are visible and documented
- ✅ **Flexibility**: Easy customization when needed
- ✅ **Guidance**: Default values and ranges provide good starting points
- ✅ **Safety**: Conservative defaults prevent misconfiguration

**Use the simple defaults for most cases, customize when you have specific needs!**
