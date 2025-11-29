# VRAM Usage Analysis for ParagonSR2 Nano Training
# RTX 3060 (11.63 GB) - Memory breakdown

## Current Configuration:
- **lq_size**: 128 (HR crop: 256x256)
- **batch_size_per_gpu**: 16
- **Model**: ParagonSR2 Nano (lightweight)
- **Dataset**: 147,353 images (147k)
- **AMP**: Enabled (fp16/bf16 training)

## Memory Breakdown Analysis:

### 1. **Model Parameters** (~50 MB)
- ParagonSR2 Nano: ~2-4M parameters
- FP32 parameters: ~8-16 MB
- FP16 parameters: ~4-8 MB (main copy)
- Optimizer states (AdamW): ~32-64 MB (2x parameters)

### 2. **Activations & Feature Maps** (~3-6 GB)
For batch_size=16, lq_size=128:
- **Input batch**: 16 × 128×128×3 × 2 bytes = 1.57 MB
- **Output batch**: 16 × 256×256×3 × 2 bytes = 6.29 MB
- **Intermediate features**:
  - Encoder: 16×64×64×256 × 2 bytes = 8.39 MB
  - Multiple attention blocks: ~50-100 MB
  - Decoder: Similar size to encoder
- **Total per sample**: ~100-200 MB
- **Batch total**: 16 × 200 MB = **3.2 GB**

### 3. **Gradients** (~50 MB)
- Same size as parameters
- AdamW needs 2 gradient states per parameter
- Total: ~100 MB

### 4. **Data Loading** (~500 MB - 2 GB)
- Python data workers: 6-8 workers
- Each worker loads: 32-64 images
- Image cache: ~50-100 MB per worker
- Total: ~500 MB - 2 GB

### 5. **AMP Memory** (~1-2 GB)
- FP32 master weights for stability
- FP16 computation buffers
- Scale management for mixed precision

### 6. **Dataset Cache** (Variable)
- Training set: 147,353 × average_image_size
- If cached: Could be several GB

### 7. **Validation Data** (~50 MB)
- Validation images loaded
- Temporary storage

## **Total Estimated Usage**:
- **Conservative**: 6-8 GB
- **Typical**: 8-11 GB
- **Peak usage**: 11+ GB (exceeds 11.63 GB limit)

## **Why VRAM Runs Out:**

1. **Activation Memory**: The biggest culprit - modern SISR models have large feature maps
2. **Data Loading**: Multiple workers loading large images simultaneously
3. **AMP Overhead**: Mixed precision adds memory overhead
4. **Large Dataset**: 147k images, even if streamed, adds system memory pressure
5. **Gradient Checkpointing**: Not used, but would increase computation

## **Optimization Strategy:**

1. **Reduce input size**: 128→96 or 128→80
2. **Reduce batch size**: 16→12 or 16→8
3. **Use gradient accumulation**: Keep effective batch size
4. **Reduce data workers**: 6→4 or 6→2
5. **Disable image rotation**: save_memory=True
6. **Optimize data loading**: Smaller prefetch

## **Memory-Safe Configuration:**

```yaml
lq_size: 96                    # Reduce from 128
batch_size_per_gpu: 8         # Reduce from 16
num_worker_per_gpu: 4         # Reduce from 6
use_rot: false                # Disable rotation
accum_iter: 2                 # Gradient accumulation
```

This should reduce memory from ~11+ GB to ~7-9 GB, fitting within RTX 3060 limits.
