# Three-Stage CC0 Dataset Filtering Pipeline

**Author**: Philip Hofmann
**Version**: 1.0.0
**License**: CC0 (Public Domain)
**Purpose**: Create publication-quality training datasets for Super-Resolution model training

---

## 🎯 **Overview**

This repository contains a sophisticated **three-stage filtering pipeline** designed to create high-quality, legally clean training datasets for Super-Resolution (SISR) models. The pipeline processes raw CC0 (public domain) images through multiple quality gates to ensure only the highest quality images reach training.

### **Key Innovation**
Unlike traditional approaches that rely on human aesthetic preferences, this pipeline uses a **technical-first approach** to avoid bias:

1. **Stage 1**: Fast algorithmic filtering (CPU) + BRISQUE quality assessment (GPU)
2. **Stage 2**: ARNIQA technical quality assessment (CNN)
3. **Stage 3**: NIMA aesthetic quality assessment (CNN)

---

## 🏗️ **Pipeline Architecture**

```
Raw CC0 Images → Stage 1 (Pre-filter) → Stage 2 (Technical) → Stage 3 (Aesthetic) → Final Dataset
     (147K+)        (Candidate tiles)     (Technical clean)      (Aesthetic quality)
```

### **Stage 1: "Sledgehammer" - High-Speed Pre-filtering**
- **Purpose**: Aggressive technical flaw removal with maximum throughput
- **Methods**: Multi-scale tiling + fast CPU metrics + BRISQUE GPU gatekeeper
- **Output**: Large set of candidate tiles (512×512px)

**CPU Metrics (Fast Filtering):**
- **Entropy**: Information density (removes smooth, low-content areas)
- **Contrast**: Pixel intensity spread (removes washed-out images)
- **Laplacian Variance**: Sharpness detection with dual bounds
  - Lower bound (50): Removes blur, fog, out-of-focus areas
  - Upper bound (3000): Prevents oversharpening artifacts
- **Blockiness**: JPEG compression artifact detection
- **Aliasing**: Moiré pattern and downscaling artifact detection

**GPU Quality Gate:**
- **BRISQUE Model**: Blind/Referenceless Image Spatial Quality Evaluator
- **Batch Processing**: Optimized for available GPU memory

### **Stage 2: "Lab Technician" - Technical Quality Gate**
- **Purpose**: Rigorous technical soundness assessment using deep learning
- **Method**: ARNIQA CNN model (specialized in technical flaw detection)
- **Focus**: Noise, compression, blur detection
- **Output**: Technically clean tiles with ARNIQA scores

### **Stage 3: "Art Critic" - Perceptual Quality Gate**
- **Purpose**: Final aesthetic quality assessment using human perception models
- **Method**: NIMA CNN model (1-10 aesthetic scoring)
- **Focus**: Visual appeal and naturalness
- **Output**: Final high-quality dataset ready for training

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
pip install torch torchvision pyiqa opencv-python pillow numpy tqdm
```

### **Basic Usage**

**Stage 1: Pre-filtering**
```bash
python stage1_prefilter.py input_folder output_tiles output.csv \
    --prefix "my_dataset" \
    --entropy_th 5.0 \
    --brisque_th 40.0
```

**Stage 2: Technical Quality Gate**
```bash
python stage2_technical_gate.py stage1_log.csv stage1_tiles output_tiles output.csv \
    --threshold 0.60 \
    --batch_size 96
```

**Stage 3: Aesthetic Quality Gate**
```bash
python stage3_perceptual_gate.py stage2_log.csv stage2_tiles final_tiles final.csv \
    --threshold 4.5 \
    --batch_size 96
```

### **Advanced Configuration**

Each script supports extensive configuration options:

```bash
# Stage 1: Multi-scale processing
python stage1_prefilter.py input output log.csv \
    --prefix "dataset" \
    --scales 1.0 0.75 0.5 0.25 \
    --entropy_th 5.5 \
    --oversharpen_th 2500 \
    --blockiness_th 35 \
    --brisque_th 45 \
    --max_workers 8

# Stage 2: Technical assessment tuning
python stage2_technical_gate.py log tiles output output.csv \
    --threshold 0.65 \
    --batch_size 128 \
    --device cuda:0

# Stage 3: Aesthetic filtering tuning
python stage3_perceptual_gate.py log tiles output output.csv \
    --threshold 5.0 \
    --batch_size 64 \
    --device cuda:0
```

---

## 📊 **Quality Metrics Reference**

| Stage | Method | Metric | Range | Purpose | Default Threshold |
|-------|--------|--------|-------|---------|-------------------|
| **1** | CPU Fast | Entropy | 0-8+ | Information density | 5.0 |
| **1** | CPU Fast | Contrast | 0-255 | Remove hazy images | 15.0 (lower bound) |
| **1** | CPU Fast | Laplacian | 0-∞ | Sharpness detection | 50-3000 (bounds) |
| **1** | CPU Fast | Blockiness | 0-∞ | Compression artifacts | 40.0 |
| **1** | CPU Fast | Aliasing | 0-1 | Moiré detection | 0.35 (upper bound) |
| **1** | GPU Model | BRISQUE | 0-100 | General quality | 40.0 |
| **2** | GPU Model | ARNIQA | 0-1 | Technical quality | 0.60 |
| **3** | GPU Model | NIMA | 1-10 | Aesthetic quality | 4.5 |

---

## 🎓 **Design Philosophy**

### **Why This Approach Works**

**1. Bias Avoidance**
- Traditional datasets rely on human aesthetic preferences
- Humans are biased toward sharpness and "pleasing" images
- IQA models inherit this bias from their training data
- Our pipeline focuses on **technical cleanliness first**

**2. Technical Soundness**
- Removes compression artifacts that confuse SR algorithms
- Prevents oversharpening that creates artificial patterns
- Ensures consistent information density across dataset
- Eliminates moiré patterns and aliasing artifacts

**3. Computational Efficiency**
- Fast-to-slow progression maximizes throughput
- CPU metrics handle bulk filtering
- GPU models focus on smaller, curated sets
- Batch processing leverages modern hardware

**4. Educational Value**
- Multi-scale processing ensures diverse content
- Complexity preservation for effective learning
- Balance between quality and informativity

---

## 📈 **Expected Results**

### **Typical Dataset Statistics**
- **Input**: 147K+ raw CC0 images
- **Stage 1 Output**: 80K-120K candidate tiles
- **Stage 2 Output**: 40K-80K technically clean tiles
- **Stage 3 Output**: 20K-60K final high-quality tiles

### **Quality Characteristics**
- **Technically Clean**: No compression artifacts, blur, or noise
- **Aesthetically Pleasing**: High visual appeal and naturalness
- **Information Rich**: Diverse texture and detail content
- **Artifact-Free**: Suitable for clean SR training

---

## 🔬 **Advanced Features**

### **Multi-Scale Processing**
```python
# Process same image at multiple scales for maximum yield
SCALES_CONFIG = [
    (1.0, "100"),    # Full resolution
    (0.75, "75"),    # 75% scale
    (0.5, "50"),     # 50% scale
    (0.25, "25"),    # 25% scale
    # + adaptive scaling for different aspect ratios
]
```

### **Memory-Efficient Batch Processing**
```python
# Optimized for 12GB+ GPUs
DEFAULT_BATCH_SIZE = {
    'brisque': 64,   # Stage 1 GPU gatekeeper
    'arniqa': 96,    # Stage 2 technical assessment
    'nima': 96       # Stage 3 aesthetic assessment
}
```

### **Comprehensive Logging**
```csv
# Detailed CSV logs with all quality scores
tile_path,original_image,prefix,scale,entropy,oversharpen,blockiness,aliasing,contrast,brisque
tile_001.png,photo_001.jpg,dataset,100,6.234,145.6,12.3,0.08,45.2,28.4
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

**CUDA Out of Memory**
```bash
# Reduce batch sizes
python stage2_technical_gate.py ... --batch_size 64
python stage3_perceptual_gate.py ... --batch_size 48
```

**Slow Processing**
```bash
# Increase worker processes
python stage1_prefilter.py ... --max_workers 12

# Use faster CPU metrics
# Increase GPU batch sizes for your hardware
```

**Low Output Counts**
```bash
# Relax thresholds for more output
python stage1_prefilter.py ... --entropy_th 4.5 --brisque_th 45
python stage2_technical_gate.py ... --threshold 0.50
python stage3_perceptual_gate.py ... --threshold 4.0
```

### **Performance Optimization**

**For Large Datasets (1M+ images):**
- Use SSD storage for input/output
- Ensure adequate RAM (32GB+)
- Use multiple GPUs if available
- Monitor GPU memory usage and adjust batch sizes

**For Small Datasets (<10K images):**
- Single GPU processing is sufficient
- Focus on threshold tuning over throughput
- Consider manual quality verification

---

## 📚 **Research Context**

### **Comparison with Existing Approaches**

**vs. BHI Dataset**
- **BHI**: Complexity-based filtering focus
- **Our Pipeline**: Technical + aesthetic quality focus
- **Impact**: Different learning characteristics, both valid approaches

**vs. Traditional Curation**
- **Traditional**: Manual selection based on human preferences
- **Our Pipeline**: Algorithmic filtering with minimal bias
- **Advantage**: Scalable, consistent, and reproducible

### **Scientific Validation**
- Uses established IQA models (BRISQUE, ARNIQA, NIMA)
- Follows computer vision best practices
- Validated through extensive testing
- Produces training-ready datasets

---

## 🤝 **Contributing**

This pipeline is designed to be extensible:

1. **Add new quality metrics**: Extend CPU/GPU metric functions
2. **Custom filtering logic**: Modify threshold processing
3. **Alternative IQA models**: Replace BRISQUE/ARNIQA/NIMA
4. **New stages**: Add additional filtering stages

---

## 📄 **License**

**CC0 (Public Domain)**
You can use, modify, and distribute these scripts freely for any purpose, including commercial applications.

---

## 🙏 **Acknowledgments**

- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator
- **ARNIQA**: An Efficient General-purpose No-reference Image Quality Assessment CNN
- **NIMA**: Neural Image Assessment model
- **PyIQa**: Python Image Quality Assessment library

---

**Citation**:
If you use this pipeline in research, please cite:
```
Philip Hofmann. "Three-Stage CC0 Dataset Filtering Pipeline for Super-Resolution Training." 2024.
```

**Contact**:
For questions or contributions, please open an issue or pull request.

---

*This pipeline represents a state-of-the-art approach to dataset curation, combining traditional image processing with modern deep learning quality assessment for optimal training data preparation.*
