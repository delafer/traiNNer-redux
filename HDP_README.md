# Hyperion Degradation Pipeline (HDP)

The Hyperion Degradation Pipeline (HDP) is an enhanced version of the Real-ESRGAN on-the-fly (OTF) degradation pipeline, designed to simulate a wider range of real-world image degradations for training state-of-the-art image restoration models.

## Overview

HDP extends the existing Real-ESRGAN degradation system with six new degradation steps that are commonly found in real-world internet images:

1. Clean Pass-Through
2. Modern Compression Artifacts (WebP/AVIF)
3. Oversharpening Artifacts
4. Chromatic Aberration
5. Demosaicing Artifacts
6. Aliasing Artifacts

## New Degradation Steps

### 1. Clean Pass-Through

Teaches the model to not alter already-perfect images.

**Configuration:**
- `p_clean`: Probability of clean pass-through (default: 0.2)

### 2. Modern Compression Artifacts

Simulates WebP and AVIF compression artifacts, which are common on the modern web.

**Configuration:**
- `webp_prob`: Probability of applying WebP compression (default: 0.3)
- `webp_range`: Quality range for WebP compression (default: [70, 90])
- `avif_prob`: Probability of applying AVIF compression (default: 0.2)
- `avif_range`: Quality range for AVIF compression (default: [75, 95])

### 3. Oversharpening Artifact Simulation

Simulates the distinctive light/dark "halos" and "ringing" caused by aggressive sharpening filters.

**Configuration:**
- `oversharpen_prob`: Probability of applying oversharpening (default: 0.4)
- `oversharpen_strength`: Strength range for oversharpening (default: [1.2, 2.5])

### 4. Chromatic Aberration Simulation

Simulates colored fringing (purple/green edges) from camera lenses.

**Configuration:**
- `chromatic_aberration_prob`: Probability of applying chromatic aberration (default: 0.3)

### 5. Demosaicing Artifact Simulation

Simulates artifacts like color moiré and zipper patterns that arise from reconstructing a color image from a camera's Bayer sensor.

**Configuration:**
- `demosaic_prob`: Probability of applying demosaicing artifacts (default: 0.2)

### 6. Aliasing Artifact Simulation

Explicitly creates "jaggies" (stair-step lines) and Moiré patterns caused by poor image resizing.

**Configuration:**
- `aliasing_prob`: Probability of applying aliasing artifacts (default: 0.3)
- `aliasing_scale_range`: Scale range for aliasing (default: [0.6, 0.9])

## Order of Operations

The new degradation functions are integrated in a logical sequence that mimics a real-world pipeline:

1. Sensor/Lens artifacts (Demosaic, CA, Aliasing)
2. Processing artifacts (Oversharpening)
3. Final Compression (JPEG/WebP/AVIF)

## Usage

To use the Hyperion Degradation Pipeline, simply enable `high_order_degradation: true` in your training configuration and add the new parameters as shown in the example configuration file.

See `example_hyperion_degradation_pipeline.yml` for a complete configuration example.

## Requirements

- PyTorch
- Pillow (with pillow-avif-plugin for AVIF support)
- OpenCV (for demosaicing artifacts)

## Implementation Details

The HDP is implemented in `traiNNer/models/realesrgan_model.py` with individual helper methods for each degradation type. All new functions are seamlessly integrated into the existing degradation methods and are individually controllable via the YAML configuration.

## Dependencies

The Hyperion Degradation Pipeline requires the following additional dependencies:

- `pillow-avif-plugin` for AVIF compression support
- `opencv-python` for demosaicing artifacts (already included in the project)

These dependencies have been added to both `pyproject.toml` and `install.sh` for automatic installation.
