# ParagonSR2

ParagonSR2 is designed as a practical "no-nonsense" super-resolution system for real use, not just benchmarks.

At a glance, ParagonSR2 aims to:
- deliver high-quality, visually pleasing outputs that hold up under close inspection,
- run fast enough for real-world deployment on consumer GPUs and modern CPUs,
- remain robust across mixed and imperfect inputs (compression, noise, sharpening, mild blur),
- integrate cleanly into production pipelines:
  - dynamic-shape ONNX export,
  - TensorRT compatibility,
  - safe FP16/INT8 post-training quantization,
  - deterministic, easily debuggable behavior.

In other words:
- ParagonSR2 should feel like a reliable tool you can ship to end users:
  - capable enough to handle complex degradations,
  - efficient enough to run interactively,
  - predictable enough that improvements come from deliberate design, not accidents.

ParagonSR2 is a practical, deployment-focused super-resolution architecture designed by Philip Hofmann and implemented for traiNNer-redux.

The goals are simple:
- High perceptual quality that looks good to real people, not just metrics.
- Clean, low-artifact outputs (no grid noise, no heavy halos, no watercolor GAN mess).
- Strong robustness to realistic degradations (compression, noise, mixed sources).
- Fully deployment-ready:
  - fuseable convs,
  - ONNX/TensorRT friendly,
  - FP16 and INT8 post-training quantization friendly,
  - dynamic shape capable.

This document explains:
- The core design decisions (Magic upsampling, LR backbone, HR refinement).
- Why ParagonSR2 does not use PixelShuffle.
- How upsampler_alpha and hr_blocks work.
- How to train and deploy ParagonSR2 reliably.

---

## High-level architecture

At a high level, ParagonSR2 does this:

1. Extract features in LR space.
2. Apply a strong but well-controlled upsampler based on Magic Kernel Sharp 2021.
3. Run a small HR refinement head to clean up remaining artifacts.
4. Output the final SR image.

In code (`network_g.type: paragonsr2_*`):

- LR backbone:
  - `conv_in` for shallow features.
  - Multiple `ResidualGroupV2` groups.
    - Each group contains `ParagonBlockV2` blocks:
      - Inception-style depthwise conv (`InceptionDWConv2d`) for multi-scale context.
      - `DynamicTransformer` with learnable depthwise kernels + LayerScale.
    - No batch/instance/group norm in the main path:
      - better stability for quantization and deployment,
      - simpler, more deterministic behavior.

- Upsampling:
  - `MagicKernelSharp2021Upsample` (from `traiNNer/archs/resampler.py`):
    - Uses the Magic Kernel Sharp 2021 recipe to upsample features:
      - a sharp, well-behaved resampling kernel,
      - designed to provide crisp detail without the grid artifacts or strong checkerboarding common in naive PixelShuffle setups.
    - Controlled by `upsampler_alpha`:
      - 0.0 = no extra sharpening (smooth baseline),
      - 1.0 = full sharp behavior,
      - default = 0.5 (chosen as a safe, crisp, low-ringing default).

- HR refinement head:
  - After Magic upsampling, ParagonSR2 applies a tiny residual CNN head:
    - `hr_conv_in`: 3x3 conv on upsampled features.
    - `hr_head`: sequence of lightweight `ResidualBlock`s (Conv + LeakyReLU + Conv + skip).
    - `conv_out`: final 3x3 conv to RGB.
  - This HR head:
    - sees the actually upsampled image representation,
    - learns to correct:
      - ghost lines near edges,
      - subtle ringing,
      - jagged diagonals,
      - small color/contrast imperfections,
    - while preserving the crispness from the Magic kernel.
  - It is intentionally small:
    - a “corrector”, not a second heavy backbone,
    - keeps inference fast and quantization/export straightforward.

---

## Why not PixelShuffle (and why Magic + Conv)

Classic choices:

- Nearest + Conv:
  - Fast, but blocky and visually crude.
- Bilinear + Conv:
  - Smooth and stable, but loses micro-contrast and fine details.
- Bicubic/Lanczos:
  - Reasonable, but still can halo and are not tailored to your network.
- PixelShuffle:
  - Popular, but:
    - can produce grid / checkerboard artifacts if misused,
    - makes some deployment/quantization scenarios more fragile.

ParagonSR2 uses a Magic-kernel-based upsampler because:

- Magic Kernel Sharp 2021:
  - Designed to be sharp, clean, and with reduced aliasing compared to naive cubic/lanczos setups.
  - Produces visually pleasing edges without heavy-handed ringing when used correctly.
- Combined with:
  - a LR backbone that learns a good feature representation,
  - a HR refinement head that explicitly cleans residual artifacts,
  - and a moderated `upsampler_alpha` (default 0.5),
- this gives:
  - sharp results that feel “high quality”,
  - with significantly fewer aliasing/halo issues than naive sharp upsamplers.

## DySample vs Magic+Conv (why this design)

Dynamic upsampling operators such as DySample were also considered during design:
- They are content-adaptive and can be very expressive for SR.
- However, they historically made:
  - clean dynamic-shape ONNX export,
  - robust TensorRT integration,
  - and predictable INT8 quantization
  more difficult, especially across different toolchains.

ParagonSR2 instead uses a Magic Kernel Sharp 2021 based upsampler implemented purely
with standard convolution-style operations:
- fully compatible with `torch.onnx.export`, ONNX Runtime, TensorRT,
- easy to fuse and optimize,
- stable under FP16 and INT8 post-training quantization.

Combined with:
- a strong LR backbone,
- the `upsampler_alpha` control (default 0.5 for sharp but safe behavior),
- and the HR residual refinement head,

this "Magic+Conv" path provides:
- sharp, alias-aware upsampling,
- low risk of grid/checkerboard artifacts,
- and a deployment graph that is simple and reliable.

---

## How ParagonSR2 builds on ParagonSR

ParagonSR2 is the natural evolution of the original ParagonSR architecture. It keeps
the core ideas that worked well and refines areas that matter for modern usage:
artifact control, deployment robustness, and clarity of behavior.

Key improvements over ParagonSR:

1. Upsampling and artifact handling

- ParagonSR:
  - Migrated from PixelShuffle to a MagicKernelSharp2021-based "Magic-Conv" upsampler.
  - Already reduced many PixelShuffle/grid artifacts.
  - Used a simple conv stack after Magic; any residual halos/jaggies were handled implicitly.

- ParagonSR2:
  - Keeps MagicKernelSharp2021 at the core, but:
    - introduces `upsampler_alpha` with a safe default of 0.5 (sharp yet controlled),
    - adds an explicit HR residual refinement head:
      - sees the upsampled features,
      - cleans halos, ghost lines, and jagged diagonals directly.
  - This makes edge behavior more predictable and easier to tune.

2. Blocks and normalization

- ParagonSR:
  - Uses GroupNorm inside ParagonBlock.
  - Effective, but normalization in the main path:
    - adds overhead,
    - can interact with quantization and color dynamics.

- ParagonSR2:
  - Core blocks are normalization-free in the main signal path:
    - rely on residual connections, LayerScale, depthwise/Inception-style convs,
      and DynamicTransformer-style mixing.
  - This:
    - simplifies numerical behavior,
    - improves INT8-friendliness,
    - reduces configuration sensitivity.

3. Explicit HR refinement stage

- ParagonSR:
  - Primarily LR feature extraction + Magic upsample + simple conv tail.

- ParagonSR2:
  - LR backbone:
    - handles denoising/deblocking and structural reasoning in LR space.
  - Magic upsampler:
    - performs the main upscaling with a principled, sharp kernel.
  - HR residual head:
    - is a dedicated, lightweight corrector:
      - designed to adjust, not overwrite, the Magic upsample result.
## Integrated two-stage refinement (built-in 1x restorer idea)

A common practical idea for high-quality SR is a two-stage pipeline:

1. Train a strong 2x SISR model.
2. Run it on LR inputs to produce enhanced-but-imperfect outputs.
3. Train a separate 1x restoration model:
   - input: the 2x model outputs,
   - target: clean HR,
   so the 1x model learns to remove artifacts and polish details left by the first stage.

ParagonSR2 embeds this concept directly into a single architecture:

- The LR backbone:
  - plays the role of the primary SISR model,
  - learns to reconstruct structure and detail efficiently in low-resolution space.
- The Magic Kernel Sharp 2021 upsampler:
  - lifts features to the target resolution with a principled, sharp, alias-aware kernel.
- The HR residual refinement head:
  - behaves like an integrated 1x restoration stage:
    - it only sees the upsampled representation,
    - it is trained end-to-end to:
      - clean halos and ringing,
      - fix subtle artifacts,
      - refine micro-texture and edges,
    - without overhauling the entire image.

This design gives you the benefits of a two-stage SR→restorer pipeline:
- artifact correction is explicit and learnable,
- complexity stays low (a small HR head, not a whole second network),
- everything is trained jointly and exported as one compact, deployment-ready model.

  - This three-step structure:
    - aligns with how high-quality SR is used in practice:
      - reconstruct → upscale → polish.

4. Deployment story

Both ParagonSR and ParagonSR2 are built with fusion in mind. ParagonSR2 tightens this:

- Uses only standard conv/activation/add operations in all critical paths.
- DynamicTransformer provides a clear dynamic→static kernel path.
- `fuse_for_release()` in ParagonSR2:
  - turns the training-time graph into a clean static conv net.
- `scripts/paragonsr2/paragonsr2_deploy.py`:
  - loads, fuses, verifies,
  - exports FP32/FP16 ONNX,
  - optionally runs INT8 calibration.

Summary:
- ParagonSR2 keeps the spirit of ParagonSR but:
  - adopts a safer, configurable Magic upsampler,
  - introduces an explicit HR refinement head,
  - removes main-path normalization,
  - and standardizes on export-friendly ops,
  making it a sharper, cleaner, and more deployment-ready successor.
The upsampler and HR head are standard conv-based operations:
- easy to understand,
- easy to fuse,
- robust under FP16 and INT8.

---

## upsampler_alpha: what it is and what we ship

`upsampler_alpha` controls how aggressively the MagicKernelSharp2021 upsampler sharpens:

- 0.0:
  - purely smooth Magic kernel (safer, softer).
- 0.5 (default):
  - balanced:
    - crisp edges,
    - low ringing risk,
    - empirically stable for L1 + modern perceptual training.
- 1.0:
  - maximum sharpness:
    - can look very crisp,
    - but combined with naive L1 can lead to overshoot/ghost edges if not controlled.

Design choice:

- All official ParagonSR2 variants default to:
  - `upsampler_alpha = 0.5`
- This means:
  - if you just set `type: paragonsr2_nano` (or tiny/s/m/l/xl) without overrides,
  - you get the safe, recommended behavior.
- Advanced users:
  - can override in YAML:
    - `network_g.upsampler_alpha: 0.3` or `0.7` etc.
  - but should understand that higher values require more careful loss tuning.

---

## HR refinement head (hr_blocks): small corrector, big impact

After Magic upsampling, ParagonSR2 applies a small residual head in HR space.

Why:
- Without it:
  - all responsibility for avoiding artifacts is pushed into:
    - LR backbone weights,
    - fixed upsampler behavior.
  - That works, but is brittle near extremely hard edges, diagonals, and content with mixed degradations.
- With a small HR residual head:
  - the network can:
    - see the full-resolution structure,
    - fix “whatever is left”:
      - subtle ringing,
      - slight aliasing,
      - micro color/contrast issues,
    - with highly local, stable adjustments.

Implementation:
- `ResidualBlock`:
  - Conv3x3 → LeakyReLU → Conv3x3 + skip.
- `hr_blocks`:
  - Controls how many such blocks are used.
  - Configurable via YAML:
    - `network_g.hr_blocks`
  - Defaults per variant:
    - Nano: 1
    - Tiny: 2
    - Anime: 2
    - S: 2
    - M: 2
    - L: 3
    - XL: 3

This scaling:

- keeps Nano extremely light,
- gives larger models more refinement capacity,
- but always treats HR head as a corrector, not a second backbone.

All ops:
- remain conv + add + LeakyReLU:
  - fully supported by:
    - reparameterization/fusion (see `fuse_for_release()`),
    - ONNX/TensorRT,
    - FP16/INT8 quantization.

---

## Variants overview

All exposed via `ARCH_REGISTRY`:

- `paragonsr2_nano`:
  - very small, fast,
  - recommended for testing, low-VRAM, and demonstration.
  - hr_blocks default: 1
  - upsampler_alpha default: 0.5
- `paragonsr2_tiny`
- `paragonsr2_anime`
- `paragonsr2_s`
- `paragonsr2_m`
- `paragonsr2_l`
- `paragonsr2_xl`

Each variant:
- scales:
  - feature width (`num_feat`),
  - depth (`num_groups`, `num_blocks`),
  - HR refinement depth (`hr_blocks`).
- All share:
  - the same core design:
    - LR backbone,
    - Magic upsampler,
    - HR residual head.

You can override:
- `upsampler_alpha`
- `hr_blocks`
in YAML if you know what you are doing.
For most users, defaults are recommended.

---

## Training recommendation (high level)

A typical two-stage training approach is recommended for high-quality models.

1) Clean fidelity pretrain

- Goal:
  - strong PSNR/SSIM,
  - artifact-minimized baseline,
  - no GAN/perceptual instability.
- Suggested:
  - Loss:
    - L1 as primary.
    - Optional:
      - small FFL (frequency) loss to stabilize high-frequencies,
      - very small TV-style smoothing if needed.
  - No GAN, no heavy perceptual in this stage.
  - Use `upsampler_alpha = 0.5`, and default `hr_blocks`.

2) Perceptual finetune

- Start from the clean pretrain.
- Add:
  - L1 (keep a strong fidelity anchor),
  - modern perceptual loss (e.g. ConvNeXt-based),
  - DISTS or similar,
  - optional light frequency loss,
  - optional Consistency loss for color/brightness stability,
  - very light GAN (e.g. r3gan with low weight) if you want extra “pop”.
- Keep:
  - the same architecture (same `upsampler_alpha`, `hr_blocks`),
  - the HR head will:
    - refine micro-artifacts instead of introducing new ones.

This division:
- keeps the pretrain clean,
- makes perceptual tuning interpretable,
- avoids mixing “architecture-caused” and “loss-caused” artifacts.

---

## Deployment and export

ParagonSR2 is designed for deployment:

- `scripts/paragonsr2/paragonsr2_deploy.py`:
  - Loads a chosen ParagonSR2 variant.
  - Loads checkpoint.
  - Calls `fuse_for_release()`:
    - freezes dynamic kernels (DynamicTransformer),
    - fuses reparameterizable convs into single convs.
  - Exports ONNX:
    - dynamic spatial shapes (height/width),
    - FP32 + FP16,
    - optional INT8 using ONNX Runtime calibration.
- All components:
  - are standard, statically-shaped ops once fused:
    - conv2d, add, activations,
    - depthwise conv for tracked kernels.

This means:
- the same model definition used for training:
  - can be fused,
  - exported to ONNX,
  - and run efficiently on GPU/CPU/accelerators
  - with behavior that matches your validated PyTorch models.

---

## Summary

ParagonSR2 is built around a few key principles:

- Use a high-quality, well-understood upsampler (Magic Kernel Sharp 2021),
  not brittle tricks.
- Keep most capacity in LR space for efficiency, but:
  - always finish with a small HR residual head that cleans up what the upsampler + backbone might get wrong.
- Choose safe, well-justified defaults:
  - `upsampler_alpha = 0.5`,
  - modest `hr_blocks` per variant.
- Make everything:
  - easy to train,
  - easy to export,
  - and visually reliable on real-world inputs.

With the provided configs and defaults, users can:
- plug in ParagonSR2,
- train clean or perceptual models,
- and deploy them without needing to tweak internal architectural knobs or sacrifice quality.
