# ParagonSR Comprehensive Benchmarking Guide

## 🎯 Overview

This guide provides a complete workflow for benchmarking all ParagonSR variants to measure inference speed and VRAM usage for publication-quality results.

## 📁 Files Created

### Core Benchmarking Tools:
- **`benchmark_paragon.py`** - Main benchmarking script
- **`train_toy_models.py`** - Quick toy model training
- **`benchmark_guide.md`** - This guide

## 🚀 Complete Workflow

### Step 1: Create Toy Models (Fast Training)
```bash
# Train all variants for 2x and 4x scaling
for variant in tiny xs s m l xl; do
    for scale in 2 4; do
        python3 scripts/benchmarking/train_toy_models.py \
            --variant $variant \
            --scale $scale \
            --output_dir models/toy_models/ \
            --iterations 100
    done
done
```

### Step 2: Deploy Models to ONNX
```bash
# Deploy all toy models to ONNX formats
for model in models/toy_models/*.pth; do
    python3 -m scripts.paragonsr.paragon_deploy \
        --input $model \
        --output models/benchmarks/ \
        --model_variant s \
        --scale 4
done
```

### Step 3: Run Comprehensive Benchmarks
```bash
# Benchmark all models
python3 scripts/benchmarking/benchmark_paragon.py \
    --models_dir models/benchmarks/ \
    --images_dir test_images/ \
    --output results/paragon_benchmark.json \
    --variants tiny xs s m l xl \
    --scales 2 4 \
    --num_images 100
```

## 📊 What Gets Measured

### Performance Metrics:
- **Inference Time**: Average, min, max, std deviation (milliseconds)
- **VRAM Usage**: Peak GPU memory consumption (MB)
- **CPU Usage**: System resource usage
- **Multiple Formats**: PyTorch fused, ONNX FP32, ONNX FP16

### Test Scenarios:
- **Different Variants**: tiny, xs, s, m, l, xl
- **Different Scales**: 2x, 4x
- **Different Resolutions**: 512x512, 1024x1024 test images
- **Warm-up Runs**: 5 iterations before timing
- **Statistical Significance**: 50-100 inference runs per model

## 🎯 Benchmarking Strategy

### Why Toy Models?
✅ **Speed**: 100 iterations = ~5-10 minutes vs hours for real training
✅ **Functionality**: Enough to test inference pipeline
✅ **Consistency**: All variants trained identically
✅ **Focus**: Measures architecture speed, not training quality

### Model Formats to Test:
1. **Fused PyTorch** (.safetensors) - Fastest PyTorch option
2. **ONNX FP32** - Standard precision for comparison
3. **ONNX FP16** - Half precision (most common deployment format)

### Comparison Models:
Consider adding benchmarks for:
- **SPAN** - Known efficient architecture
- **RealPLKSR** - Another reference point
- **ESRGAN** - Popular baseline

## 📈 Expected Results Structure

```json
{
  "benchmark_info": {
    "timestamp": "2025-10-31T08:41:00",
    "pytorch_version": "2.0.0",
    "cuda_available": true,
    "gpu": "RTX 4090"
  },
  "models": {
    "s_2x": {
      "variant": "s",
      "scale": 2,
      "formats": {
        "fused_safetensors": {
          "avg_time_ms": 45.2,
          "memory_peak": {"gpu_used_mb": 1200}
        },
        "fp16_onnx": {
          "avg_time_ms": 32.1,
          "memory_peak": {"gpu_used_mb": 800}
        }
      }
    }
  }
}
```

## 🔧 Advanced Optimizations

### ONNX Optimization Pipeline:
```bash
# 1. Simplify ONNX model
python3 -m onnxsim models/model_fp16.onnx models/model_fp16_optimized.onnx

# 2. TensorRT optimization (if available)
trtexec --onnx=models/model_fp16.onnx --saveEngine=models/model_fp16.trt

# 3. Benchmark optimized versions
```

### Performance Tuning Tips:
- **Batch Inference**: Test both single image and batch modes
- **Precision Modes**: FP32, FP16, INT8 (if calibrated)
- **Different Hardware**: Test on various GPUs if available
- **Memory Optimization**: Profile VRAM usage patterns

## 📊 Publication-Ready Results

### What to Report:
1. **Speed Comparison Table**: All variants × scales × formats
2. **Memory Usage Chart**: VRAM requirements for each model
3. **Speed vs Quality Trade-offs**: If you have quality metrics
4. **Hardware Requirements**: Minimum VRAM for real-time inference

### Key Insights to Highlight:
- **ParagonSR-S**: Best speed/quality balance for most use cases
- **ParagonSR-Tiny**: Real-time capable on modest hardware
- **Fusion Benefits**: Speedup from reparameterization
- **ONNX Benefits**: Additional speedup from optimized inference

## 🎉 Next Steps After Benchmarking

1. **Optimize Further**: Use ONNX-Simplifier, TensorRT, or other tools
2. **Quality Testing**: If needed, train proper models for quality evaluation
3. **Publication**: Create tables, charts, and writeup
4. **Community Sharing**: Publish results and models for others to use

This benchmarking framework will give you comprehensive, publication-quality performance data for all ParagonSR variants!
