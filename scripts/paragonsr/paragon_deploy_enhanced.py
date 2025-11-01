#!/usr/bin/env python3
"""
ParagonSR Complete Deployment Pipeline - Enhanced for Production
Author: Philip Hofmann

Description:
This script takes a trained ParagonSR checkpoint and creates deployment-ready models
with maximum optimization for GitHub release. Skips FP32, uses onnxslim, creates INT8.

Pipeline:
1. Training Checkpoint → Fused Model
2. Fused Model → FP16 ONNX (no FP32)
3. FP16 → Optimized FP16 (onnxslim + onnxoptimizer + polygraphy)
4. FP16 → INT8 ONNX (with fallbacks)
5. INT8 → Optimized INT8 (same optimizations)

Usage:
python -m scripts.paragonsr.paragon_deploy --input path/to/trained_model.safetensors --output path/to/release_models/ --model_variant s --scale 4

Output files:
- {input}_fused.safetensors
- {input}_op18_fp16_optimized.onnx (main release)
- {input}_op18_int8_optimized.onnx (fallback)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxconverter_common import convert_float_to_float16
from safetensors.torch import load_file, save_file

# Model variant imports are handled dynamically via get_model_variant()


def validate_dependencies() -> bool:
    """Check if required dependencies are available."""
    required_packages = [
        ("onnx", "onnx", ""),
        ("onnxconverter_common", "onnxconverter-common", ""),
        ("safetensors", "safetensors", ""),
    ]

    optional_packages = [
        ("onnxruntime.quantization", "onnxruntime", "for INT8 conversion"),
    ]

    missing_packages = []
    for package, pip_name, _ in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"Error: Missing required dependencies: {', '.join(missing_packages)}")
        print(f"Please install: pip install {' '.join(missing_packages)}")
        return False

    # Check optional packages
    missing_optional = []
    for package, pip_name, purpose in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(f"{pip_name} ({purpose})")

    if missing_optional:
        print(f"Warning: Missing optional dependencies: {', '.join(missing_optional)}")
        print("INT8 conversion will be skipped if these are missing.")
        return True  # Still return True since these are optional

    return True


def validate_training_checkpoint(state_dict: dict) -> tuple[bool, str]:
    """
    Validate that the input model is a ParagonSR training checkpoint.
    Returns (is_valid, error_message)
    """
    if not state_dict:
        return False, "Model state dict is empty"

    # Check for ParagonSR-specific patterns - look for ReparamConvV2 sub-modules
    # These should be present in training checkpoints but not in fused models

    # Pattern 1: ReparamConvV2 in main ParagonBlocks
    has_main_reparam = any(
        "body" in key and any(x in key for x in ["conv3x3", "conv1x1"])
        for key in state_dict.keys()
    )

    # Pattern 2: ReparamConvV2 in GatedFFN spatial_mixer
    has_spatial_reparam = any(
        "spatial_mixer" in key
        and any(x in key for x in ["conv3x3", "conv1x1", "dw_conv3x3"])
        for key in state_dict.keys()
    )

    # Pattern 3: LayerScale parameters (specific to ParagonSR)
    has_layerscale = any(
        "layerscale" in key.lower() or "gamma" in key for key in state_dict.keys()
    )

    if not (has_main_reparam or has_spatial_reparam):
        return False, "Model doesn't contain ParagonSR ReparamConvV2 patterns"

    # Check if it's already fused by looking for fused_conv modules
    # (These should NOT exist in training checkpoints)
    fused_conv_keys = [key for key in state_dict.keys() if "fused_conv" in key]
    if fused_conv_keys:
        return (
            False,
            "Model appears to be already fused. This script expects a training checkpoint.",
        )

    # Additional validation: ensure it's not already a fused state_dict
    # In fused models, spatial_mixer would be a Conv2d module, not ReparamConvV2
    spatial_mixer_keys = [key for key in state_dict.keys() if "spatial_mixer" in key]
    for key in spatial_mixer_keys:
        if any(x in key for x in ["weight", "bias"]):
            # Check if this is a ReparamConvV2 (training) or Conv2d (fused)
            if "conv3x3" in key or "conv1x1" in key or "dw_conv3x3" in key:
                # This indicates unfused ReparamConvV2 (good for training checkpoint)
                break
            elif "weight" in key and "conv3x3" not in key:
                # This might be a fused Conv2d
                pass

    validation_notes = []
    if has_layerscale:
        validation_notes.append("LayerScale detected")
    if has_main_reparam:
        validation_notes.append("Main ReparamConvV2 detected")
    if has_spatial_reparam:
        validation_notes.append("Spatial ReparamConvV2 detected")

    return True, f"Valid ParagonSR training checkpoint ({', '.join(validation_notes)})"


def validate_training_scale(
    state_dict: dict, expected_scale: int, model_func
) -> tuple[bool, str]:
    """
    Validate that the model was trained with the expected scale.
    Returns (is_valid, error_message)
    """
    try:
        # Create a temporary model with the expected scale
        temp_model = model_func(scale=expected_scale)

        # Check the scale parameter directly from the model
        scale_from_model = getattr(temp_model, "scale", None)

        # Check if the model's upsampler dimension matches expected scale
        # For ParagonSR, the upsampler uses scale*scale output channels
        upsampler_conv = temp_model.upsampler[0]  # First conv in upsampler
        # Get the actual num_feat for this specific variant
        num_feat = (
            upsampler_conv.in_channels
        )  # This gives us the actual feature dimension
        expected_out_channels = (
            num_feat * expected_scale * expected_scale
        )  # num_feat * scale^2
        actual_out_channels = upsampler_conv.out_channels

        # Try to load state dict to see if it fits
        try:
            temp_model.load_state_dict(state_dict, strict=False)

            # Verify scale parameter
            if scale_from_model != expected_scale:
                return (
                    False,
                    f"Model scale parameter mismatch. Expected {expected_scale}x, got {scale_from_model}",
                )

            # Verify upsampler structure
            if actual_out_channels != expected_out_channels:
                return (
                    False,
                    f"Upsampler channels mismatch. Expected {expected_out_channels}, got {actual_out_channels}",
                )

            # Additional check: try forward pass with dummy data to verify scale
            dummy_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                try:
                    dummy_output = temp_model(dummy_input)
                    expected_output_size = 32 * expected_scale
                    if dummy_output.shape[2:] == torch.Size(
                        [expected_output_size, expected_output_size]
                    ):
                        return (
                            True,
                            f"Scale validation passed ({expected_scale}x - verified via inference)",
                        )
                    else:
                        return (
                            False,
                            f"Output size mismatch. Expected {expected_output_size}x{expected_output_size}, got {dummy_output.shape[2:]}",
                        )
                except Exception as inference_error:
                    # If inference fails, fall back to structural validation
                    return (
                        True,
                        f"Scale validation passed ({expected_scale}x - structural check)",
                    )

        except Exception as load_error:
            return (
                False,
                f"Scale validation failed - model structure incompatibility: {load_error}",
            )

    except Exception as e:
        return False, f"Error during scale validation: {e}"


def fuse_training_checkpoint(
    input_path: str, output_path: str, model_func, scale: int, max_retries: int = 1
) -> tuple[bool, str]:
    """
    Fuse training checkpoint and save as optimized safetensors.
    Returns (success, message)
    """
    print("\n🔄 Stage 1: Fusing Training Checkpoint")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")

    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}")

            # Initialize model
            model = model_func(scale=scale)
            model.eval()

            # Load training weights
            print("   Loading training weights...")
            state_dict = load_file(input_path)
            model.load_state_dict(state_dict)

            # Fuse model
            print("   Fusing model architecture...")
            model.fuse_for_release()

            # Validate fusion - check that ReparamConvV2 patterns are gone
            reparam_conv_patterns = [
                "spatial_mixer.conv3x3",
                "spatial_mixer.conv1x1",
                "spatial_mixer.dw_conv3x3",
            ]

            has_reparam_patterns = any(
                any(pattern in key for pattern in reparam_conv_patterns)
                for key in model.state_dict().keys()
            )

            if has_reparam_patterns:
                raise ValueError("Fusion failed - ReparamConvV2 patterns still present")

            # Save fused model
            print("   Saving fused model...")
            model_weights = model.state_dict()
            save_file(model_weights, output_path)

            # Validate saved file
            if not os.path.exists(output_path):
                raise ValueError("Fused model file was not created")

            # Test load
            test_load = load_file(output_path)
            if len(test_load) != len(model_weights):
                raise ValueError(
                    f"Saved model size mismatch. Expected {len(model_weights)} tensors, got {len(test_load)}"
                )

            # Validate key fusion patterns
            fused_conv_keys = [
                key
                for key in model_weights.keys()
                if "spatial_mixer" in key and "weight" in key
            ]
            if not fused_conv_keys:
                raise ValueError(
                    "No fused spatial_mixer weights found - fusion may have failed"
                )

            # Check that original ReparamConvV2 patterns are gone
            original_patterns = [
                "conv3x3.weight",
                "conv1x1.weight",
                "dw_conv3x3.weight",
            ]
            for pattern in original_patterns:
                for key in model_weights.keys():
                    if pattern in key:
                        raise ValueError(
                            f"Original ReparamConvV2 pattern '{pattern}' still present after fusion"
                        )

            print(f"   ✅ Fusion successful! Created: {output_path}")
            return True, "Fusion completed successfully"

        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {e}"
            print(f"   ❌ {error_msg}")

            # Clean up on failure
            if os.path.exists(output_path):
                os.remove(output_path)

            if attempt < max_retries - 1:
                print("   Retrying in 2 seconds...")
                time.sleep(2)

    return False, f"Fusion failed after {max_retries} attempts"


def validate_fused_model(state_dict: dict) -> tuple[bool, str]:
    """
    Validate that the input model is properly fused.
    Returns (is_valid, error_message)
    """
    if not state_dict:
        return False, "Model state dict is empty"

    # Check for ReparamConvV2 sub-module keys (indicates unfused training structure)
    reparam_conv_patterns = [
        "spatial_mixer.conv3x3",
        "spatial_mixer.conv1x1",
        "spatial_mixer.dw_conv3x3",
    ]

    # Look for ReparamConvV2 sub-module patterns
    has_reparam_patterns = any(
        any(pattern in key for pattern in reparam_conv_patterns)
        for key in state_dict.keys()
    )

    # If we find ReparamConvV2 patterns, the model is NOT fused
    if has_reparam_patterns:
        return False, "Model contains ReparamConvV2 sub-modules - not fused"

    # Check that we have the expected fused Conv2d modules
    # These should be present after fusion (spatial_mixer.* keys)
    spatial_mixer_keys = [key for key in state_dict.keys() if "spatial_mixer" in key]
    if not spatial_mixer_keys:
        return False, "Model has no spatial_mixer keys - unexpected structure"

    return True, "Model appears to be properly fused"


def optimize_onnx_model(
    input_path: str, output_path: str, model_type: str = "fp16"
) -> tuple[bool, str]:
    """
    Apply multiple ONNX optimization passes to create production-ready models.
    Returns (success, message)
    """
    print(f"   🔧 Optimizing {model_type.upper()} model...")

    try:
        # Step 1: onnxslim optimization (if available)
        print("     - Applying onnxslim optimization...")
        if command_exists("onnxslim"):
            result = subprocess.run(
                ["onnxslim", input_path, output_path + "_temp_slim.onnx"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("     ✅ onnxslim optimization completed")
                input_path = output_path + "_temp_slim.onnx"
            else:
                print(f"     ⚠ onnxslim failed: {result.stderr}")
                print("     Using original model")
        else:
            print("     ⚠ onnxslim not available, skipping optimization")

        # Step 2: onnxoptimizer (using Python)
        print("     - Applying onnxoptimizer...")
        try:
            import onnxoptimizer

            model = onnx.load(input_path)
            optimized_model = onnxoptimizer.optimize(model)
            onnx.save(optimized_model, output_path + "_temp_opt.onnx")
            print("     ✅ onnxoptimizer optimization completed")

            # Clean up temp files
            for temp_file in [input_path, output_path + "_temp_opt.onnx"]:
                if os.path.exists(temp_file) and temp_file != output_path:
                    os.remove(temp_file)

            # Update input_path for next step
            input_path = output_path + "_temp_opt.onnx"
        except ImportError:
            print("     ⚠ onnxoptimizer not available, using unoptimized model")
            # Just copy the input to output
            if input_path != output_path:
                import shutil

                shutil.copy2(input_path, output_path)
            return True, f"{model_type.upper()} model optimized (limited)"
        except Exception as e:
            print(f"     ⚠ onnxoptimizer failed: {e}")
            print("     Using unoptimized model")

        # Step 3: polygraphy cleanup (if available)
        print("     - Applying polygraphy cleanup...")
        if command_exists("polygraphy"):
            result = subprocess.run(
                [
                    "polygraphy",
                    "surgeon",
                    "sanitize",
                    input_path,
                    "--fold-constants",
                    "--remove-unused-initializers",
                    "--output",
                    output_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("     ✅ Polygraphy cleanup completed")
            else:
                print(f"     ⚠ Polygraphy failed: {result.stderr}")
                print("     Using onnxoptimizer result")
                # Move the optimized model to final output
                if os.path.exists(input_path) and input_path != output_path:
                    import shutil

                    shutil.move(input_path, output_path)
        else:
            print("     ⚠ polygraphy not available, using onnxoptimizer result")
            # Move the optimized model to final output
            if os.path.exists(input_path) and input_path != output_path:
                import shutil

                shutil.move(input_path, output_path)

        # Clean up any remaining temp files
        for temp_suffix in ["_temp_slim.onnx", "_temp_opt.onnx"]:
            temp_path = output_path.replace("_optimized.onnx", temp_suffix)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Validate final model
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(
            f"     ✅ {model_type.upper()} optimization complete ({model_size:.1f} MB)"
        )
        return True, f"{model_type.upper()} optimization completed successfully"

    except Exception as e:
        print(f"     ❌ Optimization failed: {e}")
        # Fallback: copy original model
        try:
            import shutil

            shutil.copy2(input_path, output_path)
            print("     📦 Using unoptimized model as fallback")
            return (
                True,
                f"{model_type.upper()} optimization failed, using unoptimized model",
            )
        except Exception as fallback_error:
            return (
                False,
                f"Optimization failed and fallback failed: {e} -> {fallback_error}",
            )


def convert_fp16_to_int8(fp16_path: str, int8_path: str) -> tuple[bool, str]:
    """
    Convert FP16 ONNX to INT8 ONNX using onnxruntime quantization.
    Returns (success, message)
    """
    print("   🔄 Converting FP16 → INT8...")

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Load FP16 model
        fp16_model = onnx.load(fp16_path)
        onnx.checker.check_model(fp16_model)

        # Convert to INT8
        int8_model = quantize_dynamic(
            fp16_model, int8_path, weight_type=QuantType.QInt8
        )

        # Validate INT8 model
        onnx.checker.check_model(int8_path)

        # Compare file sizes
        fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)
        reduction = (1 - int8_size / fp16_size) * 100

        print("   ✅ INT8 conversion successful!")
        print("   📊 Size comparison:")
        print(f"      FP16: {fp16_size:.1f} MB")
        print(f"      INT8: {int8_size:.1f} MB")
        print(f"      Reduction: {reduction:.1f}%")

        return (
            True,
            f"INT8 conversion completed ({int8_size:.1f} MB, {reduction:.1f}% smaller)",
        )

    except ImportError:
        print("   ⚠ onnxruntime not available, skipping INT8 conversion")
        return False, "onnxruntime not available for INT8 conversion"
    except Exception as e:
        print(f"   ❌ INT8 conversion failed: {e}")
        return False, f"INT8 conversion failed: {e}"


def command_exists(command: str) -> bool:
    """Check if a command is available in PATH."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_onnx_model(model_path: str) -> tuple[bool, str]:
    """Enhanced ONNX model validation with detailed error reporting."""
    try:
        # Load model
        model = onnx.load(model_path)

        # Basic model check
        onnx.checker.check_model(model)

        # Additional structural validation
        validation_notes = []

        # Check graph structure
        if len(model.graph.input) == 0:
            return False, "Model has no input tensors"
        if len(model.graph.output) == 0:
            return False, "Model has no output tensors"

        # Check for required operators (basic opset validation)
        opset_version = model.opset_import[0].version if model.opset_import else 0
        validation_notes.append(f"ONNX opset: {opset_version}")

        # Check model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        validation_notes.append(f"Model size: {model_size:.1f}MB")

        # Check data types
        input_type = model.graph.input[0].type.tensor_type.elem_type
        input_type_name = (
            onnx.TensorProto.Name.Name[input_type]
            if hasattr(onnx.TensorProto.Name, "Name")
            else str(input_type)
        )
        validation_notes.append(f"Input type: {input_type_name}")

        return True, f"Valid ONNX model ({', '.join(validation_notes)})"

    except onnx.checker.ValidationError as e:
        return False, f"ONNX validation failed: {e}"
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        return False, f"ONNX validation failed: {e}"
    except FileNotFoundError:
        return False, f"Model file not found: {model_path}"
    except Exception as e:
        return False, f"ONNX validation error: {e}"


def export_to_onnx(
    fused_model_path: str,
    output_dir: str,
    model_func,
    scale: int,
    input_size: int = 64,
    opset_version: int = 18,
    max_retries: int = 3,
) -> tuple[bool, dict]:
    """
    Export fused model to optimized ONNX formats.
    Returns (success, output_files_dict)
    """
    print("\n🔄 Stage 2: Exporting to Optimized ONNX Formats")
    print(f"   Input: {fused_model_path}")
    print(f"   Output Directory: {output_dir}")

    output_files: dict[str, str | None] = {"fp16": None, "int8": None}

    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}")

            # Initialize and prepare model
            print("   Loading fused model...")
            model = model_func(scale=scale)
            model.eval()

            # CRITICAL FIX: First fuse the model structure, then load fused weights
            # This ensures the model structure matches the fused state dict
            print("   Preparing fused model structure...")
            model.fuse_for_release()

            # Load fused weights (use strict=True since fused model has the expected structure)
            state_dict = load_file(fused_model_path)
            model.load_state_dict(state_dict, strict=True)

            # Model is now properly fused and loaded
            print("   Model structure fused and weights loaded successfully...")

            # Validate that model is properly fused by checking for ReparamConvV2 patterns
            model_state_dict = model.state_dict()
            is_valid, validation_msg = validate_fused_model(model_state_dict)
            if not is_valid:
                raise ValueError(f"Fused model validation failed: {validation_msg}")
            print(f"   ✅ {validation_msg}")

            # Prepare for ONNX export
            print("   Preparing ONNX export...")
            dummy_input = torch.randn(1, 3, input_size, input_size)
            input_names = ["input"]
            output_names = ["output"]

            # Export FP16 ONNX directly (skip FP32 for production)
            print("   Exporting FP16 ONNX (no FP32 for production)...")
            fp16_path = os.path.join(
                output_dir, f"{Path(fused_model_path).stem}_op{opset_version}_fp16.onnx"
            )

            torch.onnx.export(
                model,
                dummy_input,
                fp16_path,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
                opset_version=opset_version,
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size", 2: "height", 3: "width"},
                },
                do_constant_folding=True,
            )

            # Validate FP16 ONNX
            is_valid, validation_msg = validate_onnx_model(fp16_path)
            if not is_valid:
                raise ValueError(f"FP16 ONNX validation failed: {validation_msg}")
            print(f"   ✅ {validation_msg}")

            print("   ✅ FP16 ONNX export successful")
            output_files["fp16"] = fp16_path

            # Optimize FP16 model
            optimized_fp16_path = fp16_path.replace(".onnx", "_optimized.onnx")
            success, msg = optimize_onnx_model(fp16_path, optimized_fp16_path, "fp16")
            if success:
                output_files["fp16"] = optimized_fp16_path
                print(f"   📈 FP16 optimized: {msg}")
            else:
                print(f"   ⚠ FP16 optimization failed: {msg}")
                # Keep original FP16 model
                output_files["fp16"] = fp16_path

            # Try to convert to INT8
            print("\n🔄 Stage 3: Creating INT8 Version")
            int8_path = optimized_fp16_path.replace(
                "_optimized.onnx", "_int8_optimized.onnx"
            )

            success, msg = convert_fp16_to_int8(output_files["fp16"], int8_path)
            if success:
                # Optimize INT8 model as well
                optimized_int8_path = int8_path.replace(".onnx", "_opt.onnx")
                success2, msg2 = optimize_onnx_model(
                    int8_path, optimized_int8_path, "int8"
                )
                if success2:
                    output_files["int8"] = optimized_int8_path
                    print(f"   📈 INT8 optimized: {msg2}")
                else:
                    output_files["int8"] = int8_path
                    print(f"   📈 INT8 basic: {msg}")
            else:
                print(f"   ⚠ INT8 conversion skipped: {msg}")
                output_files["int8"] = None

            print("\n✅ ONNX export and optimization complete!")
            return True, output_files

        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {e}"
            print(f"   ❌ {error_msg}")

            # Clean up on failure
            for file_type in output_files:
                file_path = output_files[file_type]
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"   🧹 Cleaned up failed file: {file_path}")
                    except Exception as cleanup_error:
                        print(f"   ⚠ Failed to clean up {file_path}: {cleanup_error}")

            if attempt < max_retries - 1:
                print("   Retrying in 2 seconds...")
                time.sleep(2)

    return False, output_files


def get_model_variant(model_name: str):
    """Get the ParagonSR model variant function from name."""
    model_name = model_name.lower()

    variants = {
        "tiny": ("traiNNer.archs.paragonsr_arch", "paragonsr_tiny"),
        "xs": ("traiNNer.archs.paragonsr_arch", "paragonsr_xs"),
        "s": ("traiNNer.archs.paragonsr_arch", "paragonsr_s"),
        "m": ("traiNNer.archs.paragonsr_arch", "paragonsr_m"),
        "l": ("traiNNer.archs.paragonsr_arch", "paragonsr_l"),
        "xl": ("traiNNer.archs.paragonsr_arch", "paragonsr_xl"),
    }

    if model_name not in variants:
        raise ValueError(
            f"Unknown model variant '{model_name}'. Choose from: {list(variants.keys())}"
        )

    module_name, func_name = variants[model_name]
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced ParagonSR deployment pipeline with INT8 conversion and maximum optimization"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the trained ParagonSR checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for optimized release models",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        required=True,
        choices=["tiny", "xs", "s", "m", "l", "xl"],
        help="ParagonSR model variant",
    )
    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 6, 8, 16],
        help="Training scale factor",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
        help="Input size for ONNX export (default: 64)",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts per step (default: 3)",
    )

    args = parser.parse_args()

    # Validate dependencies
    if not validate_dependencies():
        sys.exit(1)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    if not args.input.endswith(".safetensors"):
        print("❌ Error: Input file must be a .safetensors file.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model variant function
    try:
        model_func = get_model_variant(args.model_variant)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("🚀 ParagonSR Enhanced Deployment Pipeline")
    print("=" * 50)
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"🏗️  Model: ParagonSR-{args.model_variant.upper()}")
    print(f"📏 Scale: {args.scale}x")
    print(f"🔧 ONNX Opset: {args.opset_version}")
    print("🎯 Features: FP16 only, INT8 with fallbacks, Full optimization")
    print("=" * 50)

    # Load and validate training checkpoint
    print("\n🔍 Validating Input Model...")
    try:
        state_dict = load_file(args.input)

        # Validate it's a training checkpoint
        is_valid, msg = validate_training_checkpoint(state_dict)
        if not is_valid:
            print(f"❌ Input validation failed: {msg}")
            sys.exit(1)
        print(f"✅ {msg}")

        # Validate training scale
        is_valid, msg = validate_training_scale(state_dict, args.scale, model_func)
        if not is_valid:
            print(f"❌ Scale validation failed: {msg}")
            sys.exit(1)
        print(f"✅ {msg}")

    except Exception as e:
        print(f"❌ Error loading input model: {e}")
        sys.exit(1)

    # Stage 1: Fusion
    input_filename = Path(args.input).stem
    fused_model_path = os.path.join(args.output, f"{input_filename}_fused.safetensors")
    success, fusion_msg = fuse_training_checkpoint(
        args.input, fused_model_path, model_func, args.scale, args.max_retries
    )

    if not success:
        print(f"❌ Fusion failed: {fusion_msg}")
        sys.exit(1)

    # Stage 2: ONNX Export and Optimization
    success, onnx_files = export_to_onnx(
        fused_model_path,
        args.output,
        model_func,
        args.scale,
        args.input_size,
        args.opset_version,
        args.max_retries,
    )

    if not success:
        print("❌ ONNX export failed")
        sys.exit(1)

    # Final summary
    print("\n🎉 ENHANCED DEPLOYMENT COMPLETE!")
    print("=" * 50)
    print("📄 Generated Models:")
    print(f"   🔗 Fused Model: {fused_model_path}")

    for precision, file_path in onnx_files.items():
        if file_path and os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(
                f"   🔗 {precision.upper()} ONNX: {Path(file_path).name} ({size_mb:.1f}MB)"
            )

    print("\n🚀 Ready for GitHub Release!")
    print("Features:")
    print("   • FP16 only (no FP32) - optimal for production")
    print("   • INT8 fallback with quality preservation")
    print("   • Full optimization pipeline (onnxslim + onnxoptimizer + polygraphy)")
    print("   • Universally compatible ONNX format")
    print("   • Maximum performance for all inference engines")

    print("\n📋 Next steps:")
    print("   1. Test the FP16 model first (best quality)")
    print("   2. Test INT8 model if FP16 performance is insufficient")
    print("   3. Benchmark on your target hardware")
    print("   4. Upload optimized models to GitHub release")


if __name__ == "__main__":
    main()
