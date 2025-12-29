#!/usr/bin/env python3
"""
Calculate ParagonSR2 Parameter Counts
=====================================

Instantiates each ParagonSR2 variant and calculates the total number of
parameters.
"""

import sys
from pathlib import Path

import torch

# Add repo root to path for traiNNer imports
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Mocking missing dependencies that are not needed for parameter counting
import types
import unittest.mock


def mock_module(name):
    class MockModule(types.ModuleType):
        def __init__(self, name) -> None:
            super().__init__(name)
            self._mock_attrs = {}

        def __getattr__(self, _name):
            if _name not in self._mock_attrs:
                self._mock_attrs[_name] = unittest.mock.MagicMock()
            return self._mock_attrs[_name]

    m = MockModule(name)
    m.__spec__ = unittest.mock.MagicMock()
    sys.modules[name] = m
    return m


# Aggressively mock everything that might pull in heavy dependencies
mock_module("cv2")
mock_module("cv2.typing")
mock_module("pyvips")
mock_module("yaml")
mock_module("tqdm")
mock_module("tqdm.auto")
mock_module("tqdm.contrib")
mock_module("spandrel")
mock_module("spandrel.util")
mock_module("spandrel.architectures")

# We also need to mock the registry to avoid pulling in more stuff
registry_mock = mock_module("traiNNer.utils.registry")


def mock_register(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


registry_mock.ARCH_REGISTRY.register = mock_register

# Import the architecture directly using importlib to avoid triggering package-level scans
import importlib.util

arch_path = repo_root / "traiNNer" / "archs" / "paragonsr2_arch.py"
spec = importlib.util.spec_from_file_location("paragonsr2_arch", str(arch_path))
paragonsr2_arch = importlib.util.module_from_spec(spec)
sys.modules["paragonsr2_arch"] = paragonsr2_arch
spec.loader.exec_module(paragonsr2_arch)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    variants = {
        "Realtime": paragonsr2_arch.paragonsr2_realtime,
        "Stream": paragonsr2_arch.paragonsr2_stream,
        "Photo": paragonsr2_arch.paragonsr2_photo,
        "Pro": paragonsr2_arch.paragonsr2_pro,
    }

    print(f"{'Variant':<15} | {'Parameters':>15}")
    print("-" * 33)

    results = []
    for name, factory in variants.items():
        try:
            model = factory(scale=4)
            params = count_params(model)
            print(f"{name:<15} | {params:>15,}")
            results.append((name, params))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"{name:<15} | {'Error: ' + str(e):>15}")

    print("-" * 33)
    print("\nMarkdown Table for README:")
    print("| Variant | Parameters |")
    print("|---------|------------|")
    for name, params in results:
        print(f"| {name} | {params / 1e6:.3f} M ({params:,}) |")


if __name__ == "__main__":
    main()
