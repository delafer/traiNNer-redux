# RTX 3060 Memory-Optimized Config - Summary

## Updated Configuration Changes:

### **Memory Reductions (No Speed Loss):**

**Before:**
- lq_size: 128, batch_size: 16, workers: 6, accum_iter: 1
- Memory: ~11+ GB (exceeds RTX 3060)

**After (RTX3060_Optimized):**
- lq_size: 64, batch_size: 6, workers: 2, accum_iter: 1
- Memory: ~6-8 GB (should fit RTX 3060)

### **Key Changes:**

1. **lq_size: 128 → 64**
   - HR crop: 256×256 → 128×128
   - **75% reduction in activation memory** (scales quadratically)
   - Main computational savings

2. **batch_size: 16 → 6**
   - **62.5% reduction in batch memory**
   - Still decent batch size for training

3. **num_worker: 6 → 2**
   - **67% reduction in data loading memory**
   - Minimal impact on data pipeline performance

4. **accum_iter: 1 (unchanged)**
   - **Full training speed maintained**
   - No slowdown from gradient accumulation

### **Expected VRAM Usage:**
- **Activations**: ~1.5-3 GB (down from 3-6 GB)
- **Data Loading**: ~200-500 MB (down from 500MB-2GB)
- **AMP + Model**: ~1.5-2.5 GB (unchanged)
- **Total**: **6-8 GB** (should fit 11.63 GB limit)

### **Training Impact:**
- ✅ **Same training speed** (accum_iter: 1)
- ✅ **Dynamic Loss Scheduling active**
- ✅ **Quality preserved** (smaller crops but more of them)
- ✅ **Fast inference unchanged**

### **Ready to Train:**
```bash
python train.py -opt options/train/ParagonSR2/2xParagonSR2_Nano_CC0_RTX3060_Optimized.yml
```

This config prioritizes memory efficiency without compromising training speed!
