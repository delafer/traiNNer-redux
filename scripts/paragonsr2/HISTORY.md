# ParagonSR2: Architectural Evolution 🧬

The road to the final **ParagonSR2** was paved with many experiments. This document serves as a "Dev Log" summarizing the key architectural iterations found in the `arch_tries/` folder.

> **Why keep this?** Research is non-linear. Future developers (or my future self) might find value in ideas that were discarded for this specific constraints but might work elsewhere.

---

## The Journey

### 1. `paragonsr_arch.py` (v1 Legacy)
*   **Concept**: The original generic ResNet-based SR model.
*   **Outcome**: Good baseline, but lacked the efficiency for 4K realtime upscaling on mid-range cards.

### 2. `paragonsr2_arch_version0.py` - `version2.py` (The "Vibe-Coding" Phase)
*   **Idea**: Rapid prototyping of "Transformer-Lite" blocks.
*   **Drafts**:
    *   **v0**: Heavy use of standard Multi-Head Attention (MHA). OOM'd instantly on 12GB cards.
    *   **v1**: Switched to *Window Attention* but struggled with window boundary artifacts.
    *   **v2**: Tried mixing ConvNext blocks with Swin layers. Too slow for the "Realtime" goal.

### 3. `paragonsr2_arch_version3.py` (The "Gating" Breakthrough)
*   **Innovation**: Introduced the `SimpleGateBlock` (based on NAFNet/HINet ideas).
*   **Result**: Massive speedup. Gating replaced expensive non-linearities. This became the foundation for the **Stream** variant.

### 4. `paragonsr2_arch_version4.py` (The "Dual-Path" Origin)
*   **Idea**: Split the network into "Base" (Structure) and "Detail" (Texture).
*   **Mechanism**: Used `MagicKernelSharp` for the base path to guarantee structural stability even if the network fails.
*   **Adoption**: This core philosophy defined ParagonSR2.

### 5. `paragonsr2_arch_version5.py` (Efficiency Tuning)
*   **Focus**: TensorRT Optimization.
*   **Change**: Replaced all dynamic padding and weird reshaping with standard `Unfold`/`Fold` operations or simple convolutions.
*   **Result**: The birth of `paragonsr2_photo`.

---

## Final Architecture: ParagonSR2
The final released version combines the best of **v3** (Gating), **v4** (Dual-Path), and **v5** (TRT Compatible Attention).

*   **Realtime**: Pure v5 efficiency (Nano blocks).
*   **Stream**: v3 Gating logic.
*   **Photo**: v5 + SDPA Attention.
