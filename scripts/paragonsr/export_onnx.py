#!/usr/bin/env python3
"""
ParagonSR ONNX Export Utility
Author: Philip Hofmann

Description:
This script loads the PERMANENTLY FUSED ParagonSR checkpoint (from fuse_model.py)
and exports it to both FP32 and FP16 ONNX formats optimized for inference.

Usage:
python -m scripts.paragonsr.export_onnx --input path/to/fused_model.safetensors --output_dir release_models/ --model_variant s --scale 4
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxconverter_common import convert_float_to_float16
from safetensors.torch import load_file
from traiNNer.archs.paragonsr_arch import paragonsr_s


def validate_dependencies() -> bool:
    """Check if required dependencies are available."""
    try:
        import onnx
        import onnxconverter_common

        return True
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print(
            "Please install: pip install onnx onnxruntime onnxruntime-gpu onnxconverter-common"
        )
        return False


def validate_fused_model(state_dict: dict) -> tuple[bool, str]:
    """
    Validate that the input model is properly fused.
    Returns (is_valid, error_message)
    """
    # Check for key indicators of a fused model
    if not state_dict:
        return False, "Model state dict is empty"

    # Check for fused_conv keys (indicates successful fusion)
    fused_conv_keys = [key for key in state_dict.keys() if "fused_conv" in key]
    reparam_keys = [
        key
        for key in state_dict.keys()
        if any(x in key for x in ["conv3x3", "conv1x1", "dw_conv3x3"])
    ]

    if len(fused_conv_keys) > 0 and len(reparam_keys) == 0:
        return True, "Model appears to be properly fused"
    elif len(reparam_keys) > 0:
        return (
            False,
            "Model contains unfused reparameterizable blocks. Please run fuse_model.py first.",
        )
    else:
        return False, "Model structure doesn't match expected fused ParagonSR format"


def validate_onnx_model(
    model_path: str, expected_input_shape=(1, 3, 64, 64)
) -> tuple[bool, str]:
    """
    Validate that the exported ONNX model is properly formed.
    Returns (is_valid, error_message)
    """
    try:
        # Load and check model
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        # Check input/output dimensions
        if len(model.graph.input) != 1:
            return False, f"Expected 1 input, got {len(model.graph.input)}"

        if len(model.graph.output) != 1:
            return False, f"Expected 1 output, got {len(model.graph.output)}"

        input_shape = [
            dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
        ]
        output_shape = [
            dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
        ]

        # Check if shapes are reasonable for SR task
        if len(input_shape) != 4 or len(output_shape) != 4:
            return (
                False,
                f"Invalid input/output shapes: input={input_shape}, output={output_shape}",
            )

        # Check if it's properly configured for dynamic axes
        has_dynamic_input = any(
            dim.dim_value == 0
            for dim in model.graph.input[0].type.tensor_type.shape.dim[2:]
        )
        has_dynamic_output = any(
            dim.dim_value == 0
            for dim in model.graph.output[0].type.tensor_type.shape.dim[2:]
        )

        if not (has_dynamic_input and has_dynamic_output):
            print("Warning: Model may not have proper dynamic axes configuration")

        return True, "Model validation passed"

    except onnx.checker.ValidationError as e:
        return False, f"ONNX validation failed: {e}"
    except Exception as e:
        return False, f"Error validating ONNX model: {e}"


def test_model_inference(
    model_path: str, input_shape=(1, 3, 64, 64)
) -> tuple[bool, str]:
    """
    Test if the ONNX model can perform inference.
    Returns (is_valid, error_message)
    """
    try:
        import onnxruntime as ort

        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # Load ONNX model
        session = ort.InferenceSession(model_path)

        # Run inference
        outputs = session.run(None, {"input": test_input})

        # Check output
        if len(outputs) != 1:
            return False, f"Expected 1 output, got {len(outputs)}"

        output = outputs[0]
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            return False, "Model output contains NaN or Inf values"

        if output.size == 0:
            return False, "Model produced empty output"

        return True, f"Inference test passed (output shape: {output.shape})"

    except ImportError:
        return False, "ONNX Runtime not available for inference test"
    except Exception as e:
        return False, f"Inference test failed: {e}"


def get_model_variant(model_name: str):
    """Get the ParagonSR model variant function from name."""
    model_name = model_name.lower()

    if model_name == "tiny":
        from traiNNer.archs.paragonsr_arch import paragonsr_tiny

        return paragonsr_tiny
    elif model_name == "xs":
        from traiNNer.archs.paragonsr_arch import paragonsr_xs

        return paragonsr_xs
    elif model_name == "s":
        return paragonsr_s
    elif model_name == "m":
        from traiNNer.archs.paragonsr_arch import paragonsr_m

        return paragonsr_m
    elif model_name == "l":
        from traiNNer.archs.paragonsr_arch import paragonsr_l

        return paragonsr_l
    elif model_name == "xl":
        from traiNNer.archs.paragonsr_arch import paragonsr_xl

        return paragonsr_xl
    else:
        raise ValueError(
            f"Unknown model variant '{model_name}'. Choose from: tiny, xs, s, m, l, xl"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fused ParagonSR model to optimized ONNX formats (FP32/FP16)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the fused .safetensors model file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where ONNX files will be saved",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        required=True,
        choices=["tiny", "xs", "s", "m", "l", "xl"],
        help="ParagonSR model variant (tiny, xs, s, m, l, xl)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 6, 8, 16],
        help="Scale factor for the model (1, 2, 3, 4, 6, 8, 16)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
        help="Input size for ONNX export dummy tensor (default: 64)",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=18,
        help="ONNX opset version (default: 18, recommended for Mish support)",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Skip FP16 conversion (export only FP32)",
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Test ONNX models with sample inference (requires onnxruntime)",
    )

    args = parser.parse_args()

    # Validate dependencies
    if not validate_dependencies():
        sys.exit(1)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    if not args.input.endswith(".safetensors"):
        print("Error: Input file must be a .safetensors file.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model variant function
    try:
        model_func = get_model_variant(args.model_variant)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Generate output filenames
    input_name = Path(args.input).stem
    fp32_output = output_dir / f"{input_name}_fp32.onnx"
    fp16_output = output_dir / f"{input_name}_fp16.onnx"

    print("--- Starting ParagonSR ONNX Export ---")

    # 1. Initialize the model structure
    model = model_func(scale=args.scale)
    print(
        f"Initialized '{model.__class__.__name__}' architecture with scale factor {args.scale}."
    )

    # 2. CRITICAL: We must fuse the STRUCTURE before loading the fused WEIGHTS.
    # The state_dict we are about to load only has keys for the fused_conv,
    # so the model must match that structure.
    model.fuse_for_release()
    model.eval()
    print("Model structure has been prepared for fused weights.")

    # 3. Load and validate the FUSED weights
    print(f"Loading fused weights from: {args.input}")
    try:
        state_dict = load_file(args.input)

        # Validate that this is a properly fused model
        is_valid, validation_msg = validate_fused_model(state_dict)
        if not is_valid:
            print(f"Error: Input model validation failed: {validation_msg}")
            print(
                "Please ensure you are using a model that has been processed by fuse_model.py"
            )
            sys.exit(1)
        else:
            print(f"✓ {validation_msg}")

        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 4. Prepare dummy input for ONNX export
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
    input_names = ["input"]
    output_names = ["output"]

    # 5. Export to FP32 ONNX
    print(f"\n--- Exporting to FP32 ONNX: {fp32_output} ---")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(fp32_output),
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            opset_version=args.opset_version,
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
        )
        print("FP32 export successful.")

        # Comprehensive FP32 validation
        is_valid, validation_msg = validate_onnx_model(str(fp32_output))
        if is_valid:
            print(f"✓ {validation_msg}")
        else:
            print(f"⚠ {validation_msg}")

        # Inference test if requested
        if args.test_inference:
            print("Running inference test on FP32 model...")
            is_valid, test_msg = test_model_inference(str(fp32_output))
            if is_valid:
                print(f"✓ {test_msg}")
            else:
                print(f"⚠ {test_msg}")

    except Exception as e:
        print(f"Error exporting to FP32 ONNX: {e}")
        sys.exit(1)

    # 6. Convert to FP16 ONNX (unless skipped)
    if not args.no_fp16:
        print(f"\n--- Converting to FP16 ONNX: {fp16_output} ---")
        try:
            fp32_model = onnx.load(str(fp32_output))
            fp16_model = convert_float_to_float16(fp32_model)
            onnx.save(fp16_model, str(fp16_output))
            print("FP16 conversion successful.")

            # Comprehensive FP16 validation
            is_valid, validation_msg = validate_onnx_model(str(fp16_output))
            if is_valid:
                print(f"✓ {validation_msg}")
            else:
                print(f"⚠ {validation_msg}")

            # Check FP16 model characteristics
            fp16_model = onnx.load(str(fp16_output))
            fp16_types = set()
            for tensor in fp16_model.graph.initializer:
                if tensor.data_type == onnx.TensorProto.FLOAT:
                    fp16_types.add("FLOAT")
                elif tensor.data_type == onnx.TensorProto.FLOAT16:
                    fp16_types.add("FLOAT16")

            if "FLOAT16" in fp16_types:
                print("✓ Model contains FP16 data types")
            else:
                print(
                    "⚠ Model doesn't contain FP16 data types - conversion may have failed"
                )

            # Inference test if requested
            if args.test_inference:
                print("Running inference test on FP16 model...")
                is_valid, test_msg = test_model_inference(str(fp16_output))
                if is_valid:
                    print(f"✓ {test_msg}")
                else:
                    print(f"⚠ {test_msg}")

        except Exception as e:
            print("\nWARNING: FP16 conversion failed.")
            print(f"Error details: {e}")
            print("You still have the valid FP32 model.")

    print("\n--- ONNX Export Complete! ---")
    print(f"FP32 model: {fp32_output}")
    if not args.no_fp16:
        print(f"FP16 model: {fp16_output}")
    print("\nThese models are ready for deployment with:")
    print("- ONNX Runtime")
    print("- TensorRT")
    print("- DirectML")
    print("- Other ONNX-compatible inference engines")


if __name__ == "__main__":
    main()
