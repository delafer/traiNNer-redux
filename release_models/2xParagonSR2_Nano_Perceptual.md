# 🚀 2xParagonSR2_Nano_Perceptual

**Scale**: 2×
**Type**: Lightweight perceptual SR model (Nano)
**Generator**: ParagonSR2 (by Philip Hofmann)
**Discriminator**: MUNet (by Philip Hofmann)
**Training**: 200k fidelity pretrain → 80k perceptual finetune
**License**: Apache 2.0

2xParagonSR2_Nano_Perceptual is a compact 2× model built to be usable:

- sharp without going crunchy,
- perceptual without the watercolor GAN look,
- clean on real images,
- easy to deploy (FP16, dynamic ONNX, optional INT8),
- based on design decisions that came from actually trying things and discarding what didn’t feel right.

---

## 🔍 Overview

This model is:

- my ParagonSR2 Nano backbone,
- trained with my MUNet discriminator,
- upsampling via Magic Kernel Sharp–style resize+conv (no PixelShuffle),
- guided by modern perceptual signals (ConvNeXt-based, DISTS, CLIP-like contrastive),
- stabilized by a conservative R3GAN-style adversarial term,
- fed with a filtered CC0 dataset that tries to avoid common “web sharpening junk”.

Goal:
- a realistic, clean 2× upscaler that you can drop into pipelines without babysitting it.

---

## 🧩 Architecture

### ParagonSR2 (Nano) – Upsampling Journey

I wanted an upsampler that:

- doesn’t leave rasterization / checkerboard artifacts,
- doesn’t feel like a research-only trick,
- stays friendly to ONNX export, graph fusion and quantization.

What I tried:

1. PixelShuffle
   - Common and well-supported.
   - But visually can have a subtle “rasterized / checkerboard” feel in some setups.
   - For this model I didn’t like the look, so I moved away from it.

2. Bilinear + Conv
   - Simple and ONNX-friendly.
   - But bilinear smoothing costs too much high-frequency detail for what I want here.

3. Nearest + Conv
   - Keeps more high-frequency energy.
   - But too blocky on its own; not the aesthetic I’m after.

4. Dysample / dynamic upsamplers
   - Powerful and flexible.
   - For this Nano model:
     - I wanted a very straightforward, boringly reliable dynamic ONNX path.
     - Dysample is possible but more involved to make fully seamless across toolchains.
   - So I kept it out of this release.

5. Magic Kernel Sharp 2021 + Conv (“Magic+Conv”) [final choice]
   - After looking for cleaner interpolation approaches, Magic Kernel Sharp (John Costella) style kernels were the best fit:
     - high-quality, symmetric reconstruction,
     - good edge behavior,
     - less prone to typical cheap-resizer artifacts.
   - Combined with conv, it:
     - preserves structure better than bilinear,
     - avoids nearest’s blockiness,
     - stays deterministic and deployment-friendly.

So ParagonSR2 Nano here:

- uses Magic Kernel Sharp–style resize+conv for upsampling,
- no PixelShuffle,
- no exotic runtime-only dependencies,
- designed with dynamic, optimized ONNX in mind.

### MUNet – Discriminator

MUNet is my own SR-focused discriminator:

- lightweight U-Net-style, multi-scale,
- aligned with the generator’s upsampling behavior,
- looks at texture and structure where it matters.

Reasoning:

- I wanted a discriminator that:
  - understands the kind of detail ParagonSR2 can realistically produce,
  - penalizes mush and obvious artifacts,
  - doesn’t force over-stylized “GAN art”.

---

## 🧠 Training

### Stage 1 – Fidelity (200k)

- Losses:
  - L1
  - low-weight MS-SSIM
- No GAN, no perceptual extras.
- Result:
  - clean, stable 2× baseline on CC0 val.
- EMA from here is used as init for Stage 2.

### Stage 2 – Perceptual (80k)

My iteration path:

- Classic GAN setups can look impressive but often drift into watercolor / crunchy territory.
- Classic VGG perceptual loss:
  - historically used for SR,
  - but VGG is a classifier backbone with its own biases,
  - I didn’t want this model heavily driven by that.

What I settled on:

- Keep strong fidelity anchors:
  - L1 + MS-SSIM remain,
  - so geometry and color don’t collapse.

- Modern perceptual stack:
  - ConvNeXt-Tiny-based perceptual loss (shallow/mid features):
    - follows the spirit of Johnson et al. style feature losses,
    - but with a modern backbone.
  - DISTS:
    - adds a robust structure+texture similarity signal.

- CLIP-based contrastive:
  - Uses a CLIP-like embedding space as a light contrastive signal.
  - Helps align output and ground truth in a more semantic/holistic way.
  - Kept intentionally gentle.

- Consistency loss:
  - Stabilizes brightness, chroma, local structure.
  - Reduces weird tints or local instability from perceptual/GAN terms.

- Frequency loss:
  - Controls sharpness in frequency space:
    - avoid mush,
    - avoid harsh ringing and “GAN crust”.

- R3GAN-style adversarial:
  - Inspired by stabilized adversarial SR work (ESRGAN/Real-ESRGAN era).
  - Added to my framework to get:
    - more stable training,
    - more controllable behavior than naive GAN losses.
  - For this model:
    - weight is deliberately conservative,
    - enough for micro-texture and “pop”,
    - not enough to dominate fidelity/perceptual terms.

Net effect:

- A stack built to:
  - sit on a strong supervised backbone,
  - use modern perceptual/contrastive cues,
  - apply a controlled adversarial signal,
  - and avoid the typical “GAN showpiece, bad daily driver” failure mode.

---

## 📚 Dataset – 3-Stage CC0 Filtering

All data is CC0 / public domain.

I didn’t want:
- arbitrary oversharpened, heavily compressed web images as “ground truth”.

Instead I use a three-stage pipeline:

### Stage 1 – Fast pre-filter

High-speed tiling + heuristics:

- entropy
- contrast
- oversharpen bounds (too soft / too harsh filtered out)
- blockiness
- aliasing
- BRISQUE (via pyiqa) as a quick quality gate

Goal:
- drop obvious failures:
  - heavy blur,
  - strong JPEG artifacts,
  - extreme halos,
  - broken tiles.

### Stage 2 – Technical gate (ARNIQA)

- Run ARNIQA (pyiqa) on Stage 1 survivors.
- Keep tiles above a technical threshold.

Goal:
- bias towards technically sound, artifact-reduced data.

### Stage 3 – Aesthetic gate (NIMA)

- Run NIMA (pyiqa) on Stage 2 survivors.
- Keep tiles above an aesthetic threshold.

Goal:
- favor natural, coherent, non-janky images.

Note:

- BRISQUE/ARNIQA/NIMA are not perfect or absolute.
- Some metrics and web imagery can tolerate or favor sharpened/compressed content.
- Here they’re used as practical filters:
  - to push the dataset toward cleaner, more natural supervision,
  - not as a claim of “mathematical perfection”.

---

## 🎯 LR Generation

### Training LR

Custom LR generator:

- slight Gaussian blur,
- random choice of:
  - Bicubic, Bilinear, Lanczos, Box, Nearest, MagicKernelSharp-style,
- occasional multi-step resample chains.

Intent:

- avoid baking in “only bicubic” as the one true world,
- stay in a clean → mildly realistic regime,
- this is a 2× SR model, not a heavy blind-restoration model.

### Validation LR

Deterministic:

- fixed pattern over MagicKernelSharp-style, Bicubic, Lanczos, Bilinear,
- fixed blur.

Intent:

- reproducible metrics,
- stable comparisons across runs,
- validation that reflects the training flavor without randomness.

---

## 🖼️ Visual Showcase

(Replace with your own comparisons.)

- ![2xParagonSR2_Nano_Perceptual - Example 1](path/to/example1.png)
- ![2xParagonSR2_Nano_Perceptual - Example 2](path/to/example2.png)
- ![2xParagonSR2_Nano_Perceptual - Example 3](path/to/example3.png)
- ![2xParagonSR2_Nano_Perceptual - Crop Comparison](path/to/crop_comparison.png)

Suggested content:

- thin structures, foliage, textures,
- human subjects,
- subtle gradients,
- side-by-sides with pure fidelity to show:
  - sharper,
  - cleaner,
  - no heavy GAN paint.

---

## 📦 Deployment

Planned artifacts:

- 2xParagonSR2_Nano_Perceptual_fp16_dynamic.onnx
  - FP16,
  - dynamic input shapes,
  - ONNX-optimized (e.g. onnx-slim or similar).

- 2xParagonSR2_Nano_Perceptual_int8_dynamic.onnx (if it holds up)
  - post-training calibrated INT8,
  - only released if visual difference is negligible.

Designed for:

- use in real applications (apps, games, media players),
- straightforward integration without obscure dependencies.

Attribution example (optional):

```markdown
2× image upscaling powered by 2xParagonSR2_Nano_Perceptual
by Philip Hofmann (Phhofm/models).
```

---

## 🔗 References

Core framework:

- traiNNer-redux
  https://github.com/the-database/traiNNer-redux

Magic Kernel / resampling (inspiration):

- John Costella – The Magic Kernel:
  https://johncostella.com/magic/
  https://johncostella.com/magic/mks.pdf
- Classic reconstruction filter discussions:
  - Keys – Cubic Convolution Interpolation for Digital Image Processing
    https://ieeexplore.ieee.org/document/1163711

Perceptual & backbones:

- Johnson et al. – Perceptual Losses for Real-Time Style Transfer and Super-Resolution
  https://arxiv.org/abs/1603.08155
- ConvNeXt – A ConvNet for the 2020s
  https://arxiv.org/abs/2201.03545
- CLIP – Learning Transferable Visual Models From Natural Language Supervision
  https://arxiv.org/abs/2103.00020
- DISTS – Image Quality Assessment: Unifying Structure and Texture Similarity
  https://arxiv.org/abs/2004.07728

IQA / filtering:

- BRISQUE – Blind/Referenceless Image Spatial Quality Evaluator
  https://ieeexplore.ieee.org/document/6272356
- NIMA – Neural Image Assessment
  https://arxiv.org/abs/1709.05424
- ARNIQA – ARNIQA: Learning Distortion Manifold for Image Quality Assessment
  https://arxiv.org/abs/2201.08187

Adversarial SR / artifacts context:

- Distill – Deconvolution and Checkerboard Artifacts
  https://distill.pub/2016/deconv-checkerboard/
- SRGAN – Photo-Realistic Single Image Super-Resolution Using a GAN
  https://arxiv.org/abs/1609.04802
- ESRGAN – Enhanced Super-Resolution GANs
  https://arxiv.org/abs/1809.00219
- Real-ESRGAN – Training Real-World Blind SR with Pure Synthetic Data
  https://arxiv.org/abs/2107.10833

These references are for context; 2xParagonSR2_Nano_Perceptual is its own design, informed by lessons from these works rather than being a direct clone of any single one.
