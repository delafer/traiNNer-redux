#!/usr/bin/env python3
"""
Quick test to verify ParagonSR2v2 architecture works correctly.
"""

import sys

sys.path.insert(0, ".")

import torch
from traiNNer.archs.paragonsr2v2_arch import paragonsr2v2_s

print("Testing ParagonSR2v2 architecture...")
print("-" * 60)

# Test instantiation
model = paragonsr2v2_s(scale=2)
print("✓ Model instantiated successfully")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Total params: {total_params / 1e6:.2f}M")
print(f"✓ Trainable params: {trainable_params / 1e6:.2f}M")

# Test forward pass
model.eval()
dummy_input = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    output = model(dummy_input)

print("✓ Forward pass successful")
print(f"  Input shape:  {tuple(dummy_input.shape)}")
print(f"  Output shape: {tuple(output.shape)}")

# Verify output dimensions
expected_h = dummy_input.shape[2] * 2
expected_w = dummy_input.shape[3] * 2
assert output.shape == (1, 3, expected_h, expected_w), "Output shape mismatch!"
print("✓ Output dimensions correct (2x upsampling)")

print("-" * 60)
print("All tests passed! ✓")
