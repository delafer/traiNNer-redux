# ParagonSR2 Video Workflow Guide

This guide explains how to use the **Temporal Feature-Tap** functionality for ParagonSR2. This mode improves video stability by smoothing intermediate feature maps between frames without requiring any model retraining.

## 1. Overview
*   **Standard Mode**: Outputs only the super-resolved image. Best for single images.
*   **Video Mode**: Outputs [(sr_image, feature_map)](file:///home/phips/Documents/GitHub/traiNNer-redux/scripts/paragonsr2/benchmark_release.py#286-331). The `feature_map` is used by your external player/script to blend with the previous frame's features.

## 2. Converting to ONNX (Video Mode)
Use the `--video` flag with the conversion script. This enables the secondary output.

```bash
python scripts/paragonsr2/convert_onnx_release.py \
    --input models/my_model.safetensors \
    --scale 2 \
    --arch paragonsr2_realtime \
    --fp16 \
    --dynamic \
    --video
```
**Output**: `models/paragonsr2_realtime_fp16_video.onnx`

## 3. Building TensorRT Engine
Use `trtexec` just like before. The tool automatically detects the second output (`feature_map`).

**Example for RTX 3060 (Realtime 2x):**
```bash
trtexec --onnx=models/paragonsr2_realtime_fp16_video.onnx \
    --saveEngine=models/paragonsr2_realtime_fp16_video.trt \
    --fp16 \
    --minShapes=input:1x3x360x640 \
    --optShapes=input:1x3x720x1280 \
    --maxShapes=input:1x3x1080x1920
```

## 4. Benchmarking
Use the `--video` flag to verify performance. This ensures you measure the slight overhead of writing the feature map to VRAM.

```bash
python scripts/paragonsr2/benchmark_release.py \
    --input test_images/ \
    --scale 2 \
    --pt_model models/my_model.safetensors \
    --arch paragonsr2_realtime \
    --trt_engine models/paragonsr2_realtime_fp16_video.trt \
    --video
```

## 5. Inference Logic (Python/C++ Orchestrator)
The neural network is **stateless**. You must manage the state in your application code.

**Pseudocode:**
```python
# Initialize
prev_feature = None
alpha = 0.2  # Smoothing factor (0.2 = 20% new, 80% old)

for frame in video_stream:
    # 1. Scene Cut Detection (CRITICAL)
    if is_scene_cut(frame, prev_frame):
        prev_feature = None
    
    # 2. Inference
    # Model returns (image, raw_new_feature)
    sr_image, new_feature = trt_engine.execute(frame)
    
    # 3. Temporal Smoothing
    if prev_feature is not None:
        # Blind exponential smoothing
        smoothed_feature = alpha * new_feature + (1 - alpha) * prev_feature
    else:
        smoothed_feature = new_feature
        
    # 4. Update State
    prev_feature = smoothed_feature
    
    # 5. Display/Save sr_image
    display(sr_image)
```
