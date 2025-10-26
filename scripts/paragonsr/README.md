# ParagonSR: A High-Performance Super-Resolution Architecture
*An Architecture by Philip Hofmann*

## 1. Introduction

ParagonSR is a state-of-the-art, general-purpose super-resolution architecture designed for a superior balance of **peak quality, training efficiency, and inference speed**. It represents a synthesis of cutting-edge concepts from a multitude of modern network designs, including `RealPLKSR`, `HyperionSR`, `FDAT`, `MoSRv2`, and more.

The primary goal of ParagonSR is to provide a powerful and practical solution for SISR tasks, capable of achieving state-of-the-art results while being significantly faster in real-world deployment thanks to its reparameterizable design.

## 2. Core Design Philosophy

The architecture is built on a principled, hybrid approach that combines the most effective and efficient ideas from both modern CNNs and Transformer-based models.

### The ParagonBlock: A Synergistic Core
The heart of the network is the `ParagonBlock`, a novel block designed to maximize performance and efficiency:

1.  **Efficient Multi-Scale Context (The "Eyes"):** Instead of a single large kernel, the ParagonBlock uses an **Inception-style Depthwise Convolution** (`InceptionDWConv2d`). This module captures features at multiple spatial scales (square, horizontal, and vertical) simultaneously. It's a more expressive and parameter-efficient method for gathering the wide receptive field necessary for high-quality restoration.

2.  **Powerful Gated Transformation (The "Brain"):** The features are then processed by a **Gated Feed-Forward Network** (`GatedFFN`). This is a powerful, non-linear feature transformer inspired by modern language models. Its gating mechanism (`act(g) * i`) allows the network to dynamically route and modulate information, a key technique for learning complex patterns that far exceeds the capability of simple convolutional layers.

3.  **Inference-Time Speed (The "Afterburner"):** The entire spatial mixing component of the block is built on **reparameterization**. The complex, multi-branch training-time structure is mathematically fused into a single, simple, and ultra-fast convolution during evaluation. This makes the final model ideal for ONNX export and high-speed applications like video processing.

### Hierarchical Structure
The ParagonBlocks are organized into **Residual Groups**, a proven structure that improves training stability and gradient flow. This allows for the creation of deeper, more powerful networks that are easier to train to convergence.

## 3. The ParagonSR Family: A Model for Every Need

ParagonSR comes in a variety of sizes, allowing users to choose the perfect balance of quality and performance for their hardware.

| Variant | Feature Dim | Depth (#Groups x #Blocks) | Training Target (VRAM) | Inference Target (VRAM) | Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `paragonsr_tiny`| 32 | 3 x 3 (9) | ~4-6GB | **Any GPU/CPU** | **Real-Time Video**, Previews |
| `paragonsr_xs` | 48 | 4 x 4 (16) | ~6-8GB | ~4-6GB | Low-End Hardware, Fast Images |
| `paragonsr_s` | 64 | 6 x 6 (36) | **~12GB** | ~6-8GB | **Flagship Model**, High Quality |
| `paragonsr_m` | 96 | 8 x 8 (64) | ~16-24GB | ~8-12GB | Prosumer Quality |
| `paragonsr_l` | 128 | 10 x 10 (100)| >24GB | ~12GB+ | Enthusiast/SOTA Quality |
| `paragonsr_xl` | 160 | 12 x 12 (144)| 48GB+ | >24GB | Research/Benchmark Chasing |


## 4. Usage with traiNNer-redux

### Installation
Place the `ParagonSR_arch.py` file into your `traiNNer/archs/` directory. The framework will automatically detect and register the architecture.

### Configuration
In your training `config.yaml`, specify the desired variant under the `network_g` section.

**Example for training the flagship `S` model:**
```yaml
# In your config.yaml

network_g:
  type: paragonsr_s
  # scale is automatically passed, no need to specify it here.
```

## 5. Inference Performance & ONNX Export

A key feature of ParagonSR is its reparameterizable design, which enables a significant speed-up for the final, deployed model.

-   **During Training (`model.train()`):** The network uses its full, multi-branch architecture to maximize learning capacity.
-   **During Inference (`model.eval()`):** The parallel branches of the `ReparamConv` blocks are mathematically **fused** into a single, highly optimized 3x3 convolution.

To create a release-ready model (e.g., for ONNX), you **must** perform this fusion step after training is complete.

**Conceptual Workflow for Release:**
1.  Load your final trained checkpoint (`.pth` or `.safetensors`) into the ParagonSR model structure.
2.  Call `model.eval()` on the model. This will automatically trigger the fusion logic.
3.  Save the `state_dict` of this new, fused model. This is your permanent, high-speed inference model.
4.  Export this fused model to ONNX. The resulting graph will be simple, static, and ideal for acceleration with runtimes like **ONNX Runtime** and **NVIDIA TensorRT**.
