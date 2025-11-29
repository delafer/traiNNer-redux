# Training Log Files - Complete Guide

## 🎉 **Log Files Are Already Created Automatically!**

Your traiNNer training **automatically creates comprehensive log files** with all console output, loss values, and training metrics!

## 📁 **Log File Locations**

Log files are automatically created in:
```
experiments/{config_name}/logs/train_{config_name}_{timestamp}.log
```

### **Example Locations:**
```
experiments/
├── 2xParagonSR2_Nano_BHI_Small/
│   ├── logs/train_2xParagonSR2_Nano_BHI_Small_20251129_103351.log
│   ├── models/                    # Checkpoints
│   └── training_states/          # Resume states
├── 2xParagonSR2_Nano_PSISR_D/
│   ├── logs/train_2xParagonSR2_Nano_PSISR_D_20251129_103420.log
│   └── ...
└── 2xParagonSR2_Nano_CC0_147k/
    ├── logs/train_2xParagonSR2_Nano_CC0_147k_20251129_103500.log
    └── ...
```

## 🔍 **Quick Log File Finder**

Use the provided utility to find your log files:

```bash
# List all training log files
python find_training_logs.py

# View a specific log file (replace 1 with log number)
python find_training_logs.py 1
```

## 📊 **What's Included in Log Files**

### **Training Progress:**
```
[2025-11-29 10:33:51] INFO: Training statistics for 2xParagonSR2_Nano_BHI_Small:
    Number of train images:        50,000
    Batch size per gpu:                 32
    Total iters:                   40,000
```

### **Loss Values:**
```
[2025-11-29 10:34:15] INFO: Training: [50/40000] lr: 2.00e-04,
    l_g_l1: 0.0523, l_g_ssim: 0.0089, time: 0.45s, data_time: 0.12s
```

### **Validation Metrics:**
```
[2025-11-29 10:35:00] INFO: Validation [1000/40000]:
    val/psnr: 28.45, val/ssim: 0.856
```

### **Dynamic Loss Scheduling:**
```
[2025-11-29 10:40:00] INFO: Dynamic loss scheduler baseline established
    at iteration 200. Baseline loss values: {'l_g_l1': 0.0523, 'l_g_ssim': 0.0089}
```

### **System Information:**
```
[2025-11-29 10:33:51] INFO: GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-xxx)
[2025-11-29 10:33:51] INFO: mem: 18.0 GB / 24.0 GB
[2025-11-29 10:33:51] INFO: Training with manual_seed=1024
```

### **Training Completion:**
```
[2025-11-29 11:45:30] INFO: End of training. Time consumed: 1:12:30
[2025-11-29 11:45:30] INFO: Save the latest model.
```

## 💡 **Key Benefits**

### **Complete Record:**
- **All console output** saved automatically
- **Every training iteration** tracked with loss values
- **Validation metrics** at every checkpoint
- **System information** for debugging
- **Training timing** for performance analysis

### **Future Reference:**
- **Compare training runs** across different datasets
- **Debug training issues** with detailed logs
- **Analyze convergence** patterns and timing
- **Track hyperparameter** effects on training
- **Document training** experiments

### **Analysis Capabilities:**
- **Loss curves**: Plot training loss over time
- **Validation scores**: Track PSNR/SSIM improvements
- **Training speed**: Analyze per-iteration timing
- **Dynamic weights**: See how loss scheduling adapts
- **Resource usage**: Monitor GPU memory and utilization

## 🎯 **Perfect for Dataset Comparison**

Each dataset comparison training run will have its own log file:
- **BHI Small**: `train_2xParagonSR2_Nano_BHI_Small_xxx.log`
- **PSISR-D**: `train_2xParagonSR2_Nano_PSISR_D_xxx.log`
- **CC0 147k**: `train_2xParagonSR2_Nano_CC0_147k_xxx.log`

You can easily:
1. **Compare convergence speeds** across datasets
2. **Analyze final quality** metrics
3. **Track training stability** patterns
4. **Debug any issues** that arise

## 🚀 **Ready to Use**

The log file system is **already configured and working**! When you run training with any of your dataset comparison configs, log files will automatically be created in the experiments folder with all loss values and training details preserved for future reference.

No additional configuration needed - it's all automatic!
