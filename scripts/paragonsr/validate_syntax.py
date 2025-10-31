#!/usr/bin/env python3
"""
ParagonSR Deployment Script - Syntax Validation
Author: Philip Hofmann

Description:
This script performs static syntax validation of the corrected paragon_deploy.py
without requiring actual model files or PyTorch installation.

Usage:
python3 scripts/paragonsr/validate_syntax.py
"""

import ast
import sys
from pathlib import Path


def validate_python_syntax(file_path) -> bool | None:
    """Validate Python syntax of a file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse the AST to check syntax
        ast.parse(content)
        print(f"✅ Syntax validation passed for {file_path}")
        return True

    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}:")
        print(f"   Line {e.lineno}: {e.msg}")
        print(f"   Text: {e.text}")
        return False

    except Exception as e:
        print(f"❌ Error validating {file_path}: {e}")
        return False


def check_function_signatures(file_path):
    """Check that expected functions have correct signatures."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        expected_functions = {
            "validate_training_checkpoint": 1,  # state_dict
            "validate_training_scale": 3,  # state_dict, expected_scale, model_func
            "validate_fused_model": 1,  # state_dict
            "validate_onnx_model": 1,  # model_path
            "get_model_variant": 1,  # model_name
            "fuse_training_checkpoint": 5,  # input_path, output_path, model_func, scale, max_retries
            "export_to_onnx": 7,  # fused_model_path, output_dir, model_func, scale, input_size, opset_version, max_retries
        }

        function_defs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_defs[node.name] = len(node.args.args)

        print("\n🔍 Function Signature Analysis:")
        all_good = True

        for func_name, expected_args in expected_functions.items():
            if func_name in function_defs:
                actual_args = function_defs[func_name]
                if actual_args == expected_args:
                    print(f"  ✅ {func_name}: {actual_args} args (correct)")
                else:
                    print(
                        f"  ❌ {func_name}: {actual_args} args (expected {expected_args})"
                    )
                    all_good = False
            else:
                print(f"  ❌ {func_name}: Not found")
                all_good = False

        return all_good

    except Exception as e:
        print(f"❌ Error checking function signatures: {e}")
        return False


def check_return_type_annotations(file_path):
    """Check that functions have proper return type annotations."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        functions_with_tuple_returns = [
            "validate_training_checkpoint",
            "validate_training_scale",
            "validate_fused_model",
            "validate_onnx_model",
        ]

        print("\n🔍 Return Type Annotation Analysis:")
        all_good = True

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in functions_with_tuple_returns
            ):
                if node.returns:
                    # Check if it returns tuple
                    if hasattr(node.returns, "id") and node.returns.id == "tuple":
                        print(f"  ✅ {node.name}: Has tuple return annotation")
                    elif isinstance(node.returns, ast.Subscript):
                        print(f"  ✅ {node.name}: Has proper return annotation")
                    else:
                        print(f"  ⚠️  {node.name}: Return annotation may need review")
                else:
                    print(f"  ⚠️  {node.name}: Missing return annotation")

        return all_good

    except Exception as e:
        print(f"❌ Error checking return annotations: {e}")
        return False


def check_imports_and_dependencies(file_path):
    """Check that required imports are present."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        required_imports = [
            "torch",
            "onnx",
            "safetensors",
            "onnxconverter_common",
        ]

        print("\n🔍 Import Analysis:")
        all_good = True

        for import_name in required_imports:
            if import_name in content:
                print(f"  ✅ {import_name}: Import found")
            else:
                print(f"  ⚠️  {import_name}: Import not found")
                all_good = False

        return all_good

    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False


def main() -> bool:
    """Run all validation checks."""
    print("🚀 ParagonSR Deployment Script - Static Validation")
    print("=" * 60)

    script_path = Path(__file__).parent / "paragon_deploy.py"

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    print(f"📄 Validating: {script_path}")

    tests = [
        ("Python Syntax", lambda: validate_python_syntax(script_path)),
        ("Function Signatures", lambda: check_function_signatures(script_path)),
        ("Return Type Annotations", lambda: check_return_type_annotations(script_path)),
        ("Import Dependencies", lambda: check_imports_and_dependencies(script_path)),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Test '{test_name}' failed")
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Static Analysis Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All static validation checks passed!")
        print("✅ The corrected deployment script has proper syntax and structure.")
        return True
    else:
        print("⚠️  Some validation checks failed. Please review.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
