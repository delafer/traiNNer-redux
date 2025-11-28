# CC0 Dataset Preparation: A Three-Stage Quality Pipeline

**Author**: Philip Hofmann
**Purpose**: Create publication-quality training dataset from CC0 images
**Pipeline**: 3-stage filtering with technical and perceptual quality gates

---

## 🎯 **Overview**

This document explains the sophisticated **three-stage filtering pipeline** used to create the high-quality CC0 dataset for super-resolution training. This approach combines fast algorithmic metrics with deep learning quality models to ensure only the highest quality images proceed to training.

## 🏗️ **Pipeline Architecture**

```
Raw CC0 Images → Stage 1 (Pre-filter) → Stage 2 (Technical) → Stage 3 (Aesthetic) → Final Dataset
     (147K+)        (Candidate tiles)     (Technical clean)      (Aesthetic quality)
```

Stage 1: "Sledgehammer" (Fast CPU + GPU pre-filtering)

Multi-scale tiling (100%, 75%, 50%, 25%, adaptive)
Fast metrics: entropy, contrast, Laplacian, blockiness, aliasing
BRISQUE GPU gatekeeper
Goal: Aggressive filtering with maximum throughput

Stage 2: "Lab Technician" (ARNIQA technical gate)

CNN-based technical quality assessment
Focuses on noise, compression, blur detection
Goal: Ensure technical soundness

Stage 3: "Art Critic" (NIMA aesthetic gate)

Aesthetic quality prediction (1-10 scale)

Goal: Ensure visual appeal and naturalness

🎓 Why This Approach is Brilliant:
Multi-dimensional Quality: Technical + Perceptual + Statistical
Computational Efficiency: Fast-to-slow progression
Batch Processing: Leverages GPU memory effectively
Proven Models: BRISQUE → ARNIQA → NIMA (established quality metrics)

---

## 📊 **Stage 1: High-Speed Tiling & Pre-Filtering**

**Goal**: Create large set of candidate tiles with basic quality screening

### **Multi-Scale Tiling**
- **Tile Size**: 512×512 pixels (optimal for 2× SR training)
- **Scales**: 100%, 75%, 50%, 25%, + adaptive scaling
- **Strategy**: Process same image at multiple scales for maximum tile yield

### **Fast CPU Metrics** (Sledgehammer Approach)
These metrics provide aggressive filtering with minimal computational cost:

1. **Entropy** (Information Density)
   - Removes: Smooth, low-information areas

2. **Contrast** (Pixel Intensity Spread)
   - Removes: Washed-out, hazy, low-contrast images

3. **Laplacian Variance** (Sharpness Detection)
   - Removes: Out-of-focus, blur, and artificially sharpened images
   - **Key Innovation**: Upper bound prevents oversharpening artifacts

4. **Blockiness** (JPEG Compression Artifacts)
   - Removes: Heavily compressed images
   - **Critical for SR**: Compression artifacts confuse upsampling

5. **Aliasing Score** (Moiré Pattern Detection)
   - Removes: Downscaling artifacts and interference patterns

6. **IC9600 Complexity** (Texture Detail)
   - **Purpose**: Ensures informative patches for SR training
   - **Philosophy**: Not just "clean" but "educationally valuable"

### **GPU Quality Gate**
- **BRISQUE Model**: Blind/Referenceless Image Spatial Quality Evaluator
- **Purpose**: Fast general quality assessment
- **Batch Size**: Optimized for available GPU memory
- Provides additional quality scoring alongside CPU metrics

### **Output**
- Large set of 512×512 candidate tiles
- Detailed CSV log with all quality scores
- Multi-scale coverage for maximum data diversity

---

## 🔬 **Stage 2: Technical Quality Gate**

**Goal**: Rigorous technical soundness filtering using deep learning

### **ARNIQA Model**
- **Full Name**: An Efficient General-purpose No-reference Image Quality Assessment CNN
- **Specialization**: Technical flaw detection (noise, compression, blur)
- **Batch Size**: Optimized for available GPU memory
- Provides rigorous technical quality assessment

### **Why ARNIQA?**
- **CNN-based**: Fast inference, large batch processing
- **Technical Focus**: Specifically trained for compression artifacts and noise
- **Robust**: Handles diverse degradation types
- **Efficient**: Much faster than Transformer-based alternatives

### **Filtering Strategy**
- Aggressive technical quality threshold
- Ensures only clean, sharp, artifact-free images proceed
- Removes remaining compression noise and technical imperfections

### **Output**
- Smaller set of technically sound tiles
- New CSV log with ARNIQA scores
- High confidence in technical cleanliness

---

## 🎨 **Stage 3: Perceptual Quality Gate**

**Goal**: Final aesthetic quality assessment using human perception models

### **NIMA Model**
- **Full Name**: Neural Image Assessment
- **Specialization**: Aesthetic quality prediction (1-10 scale)
- **Batch Size**: Optimized for available GPU memory
- Provides aesthetic quality scoring for naturalness

### **Why NIMA?**
- **Perceptual Focus**: Trained on human aesthetic preferences
- **Speed**: Much faster than Transformer-based alternatives (MANIQA, etc.)
- **Proven**: Widely used for aesthetic quality assessment
- **Scale**: Direct 1-10 scoring for easy interpretation

### **Aesthetic Considerations**
- Ensures final dataset is visually appealing
- Removes synthetic-looking or unnatural images
- Promotes authentic photographic content
- Proxy for "naturalness" in SR training

### **Output**
- Final high-quality CC0 dataset (thousands of tiles)
- CSV log with NIMA aesthetic scores
- Ready for super-resolution training

---

## 📈 **Quality Metrics Summary**

| Stage | Method | Metric | Purpose |
|-------|--------|--------|---------|
| **1** | CPU Fast | Entropy | Information density |
| **1** | CPU Fast | Contrast | Remove hazy/washed-out |
| **1** | CPU Fast | Laplacian | Sharpness detection (prevents oversharpening) |
| **1** | CPU Fast | Blockiness | Compression artifacts |
| **1** | CPU Fast | Aliasing | Moiré patterns |
| **1** | CPU Fast | IC9600 Complexity | Texture detail and educational value |
| **1** | GPU Model | BRISQUE | General quality assessment |
| **2** | GPU Model | ARNIQA | Technical soundness |
| **3** | GPU Model | NIMA | Aesthetic quality |

---

## 🎯 **Key Design Decisions**

### **1. Multi-Scale Approach**
- Processing same images at multiple scales maximizes tile yield
- Ensures diversity in image content and quality
- Adaptive scaling for different image aspect ratios

### **2. Fast-to-Slow Progression**
- Stage 1: Aggressive, fast CPU filtering
- Stage 2: Moderate GPU technical assessment
- Stage 3: Selective GPU perceptual assessment
- **Result**: Maximum efficiency at each stage

### **3. Multiple Quality Dimensions**
- **Technical**: Sharpness, contrast, compression
- **Perceptual**: Aesthetic appeal, naturalness
- **Statistical**: Information content, aliasing
- **Comprehensive**: Covers all major quality aspects

### **4. Batch Processing**
- Leverages GPU memory for large batch sizes
- Multiprocessing for CPU-bound operations
- Optimized for modern hardware (12GB+ GPUs)

---

## 📊 **Expected Results**

### **Dataset Statistics**
- **Input**: 147K+ raw CC0 images
- **Stage 1**: Large set of candidate tiles after aggressive pre-filtering
- **Stage 2**: Technically sound tiles after ARNIQA assessment
- **Stage 3**: Final high-quality tiles after aesthetic evaluation

### **Quality Characteristics**
- **Technically Clean**: No compression artifacts, blur, or noise
- **Aesthetically Pleasing**: High visual appeal and naturalness
- **Information Rich**: Diverse texture and detail content
- **Artifact-Free**: Suitable for clean SR training

---

## 🔍 **Comparison with BHI Dataset**

### **Key Difference: Complexity Filtering**
- **BHI**: Includes complexity-based filtering for informative patches
- **CC0**: This pipeline focuses on technical + aesthetic quality
- **Impact**: May affect training convergence and final performance

### **Research Question**
This filtering approach allows us to test:
1. **Technical + Aesthetic filtering** vs **Complexity filtering**
2. **Quality over quantity** vs **Informative content emphasis**
3. **Which approach better serves SR training**

---

## 🚀 **Usage for Training**

The final dataset is optimized for:
- **Super-Resolution Training**: Clean, high-quality ground truth
- **Fast Convergence**: High information density per tile
- **Artifact-Free Learning**: No bad examples to confuse model
- **Diverse Content**: Multiple scales and image types

This pipeline represents a **state-of-the-art approach** to dataset curation, combining traditional image processing with modern deep learning quality assessment for optimal training data preparation.
