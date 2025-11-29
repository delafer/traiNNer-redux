# VRAM Attention Optimization for Validation

## Problem Identified

Current `EfficientSelfAttention` has flawed chunked attention:
```python
attn_chunk = torch.bmm(q_chunk, k)  # Computes chunk vs ALL tokens
```

This causes OOM for large validation images (>256×256).

## Solution: Memory-Efficient Attention

### 1. Reduce Thresholds
```python
self.max_full_attention_tokens = 2048     # Was 4096
self.max_chunked_attention_tokens = 16384 # Was 65536
```

### 2. Improve Chunking
```python
chunk_size = min(256, num_tokens // 16)  # Smaller chunks
```

### 3. Spatial Hierarchical Attention
Divide 512×512+ images into 64×64 spatial chunks with overlap blending.

## Expected Results (RTX 3060)

- **128×128**: 0.5-1 GB VRAM (was 2-4 GB)
- **256×256**: 1-2 GB VRAM (was 8-16 GB)
- **512×512**: 2-4 GB VRAM (was OOM)

## Configuration
```yaml
train:
  val_batch_size: 1
  val_max_image_size: 512
  val_attention_optimization: true
```

This eliminates validation OOM while maintaining quality.
