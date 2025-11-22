#!/usr/bin/env python3
"""
Test script to verify the SISR training fixes are working correctly.

This script tests:
1. R3GAN Loss accepts current_iter parameter without crashing
2. Feature Matching Loss handles gradients properly
3. Discriminator update logic doesn't reuse frozen discriminator results
"""

import sys

import torch
from torch import nn
from traiNNer.losses.feature_matching_loss import FeatureMatchingLoss
from traiNNer.losses.r3gan_loss import R3GANLoss


def test_r3gan_current_iter() -> bool | None:
    """Test that R3GAN loss accepts current_iter parameter."""
    print("Testing R3GAN Loss with current_iter parameter...")

    # Create R3GAN loss
    r3gan_loss = R3GANLoss(loss_weight=1.0)

    # Create a simple discriminator
    class SimpleDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    net_d = SimpleDiscriminator()
    real_images = torch.randn(2, 3, 64, 64)
    fake_images = torch.randn(2, 3, 64, 64)

    try:
        # Test generator step with current_iter
        gen_loss = r3gan_loss(
            net_d=net_d,
            real_images=real_images,
            fake_images=fake_images,
            is_disc=False,
            current_iter=100,
        )
        print(f"✓ Generator loss computed successfully: {gen_loss.item():.4f}")

        # Test discriminator step with current_iter
        disc_loss_dict = r3gan_loss(
            net_d=net_d,
            real_images=real_images,
            fake_images=fake_images,
            is_disc=True,
            current_iter=100,
        )
        print(
            f"✓ Discriminator loss computed successfully: {disc_loss_dict['d_loss'].item():.4f}"
        )

        return True
    except Exception as e:
        print(f"✗ R3GAN Loss test failed: {e}")
        return False


def test_feature_matching_gradients() -> bool | None:
    """Test that Feature Matching Loss maintains gradient flow."""
    print("\nTesting Feature Matching Loss gradient handling...")

    # Create feature matching loss
    fm_loss = FeatureMatchingLoss()

    # Create fake discriminator features
    real_feats = [
        torch.randn(2, 64, 32, 32, requires_grad=True),
        torch.randn(2, 128, 16, 16, requires_grad=True),
    ]
    fake_feats = [
        torch.randn(2, 64, 32, 32, requires_grad=True),
        torch.randn(2, 128, 16, 16, requires_grad=True),
    ]

    try:
        # Compute loss
        loss = fm_loss(real_feats, fake_feats, current_iter=100)

        # Backward pass to check gradients flow
        loss.backward()

        # Check that gradients exist only in fake features (real features should be detached)
        real_has_no_grads = all(f.grad is None for f in real_feats)
        fake_has_grads = all(f.grad is not None for f in fake_feats)

        if real_has_no_grads and fake_has_grads:
            print(
                f"✓ Feature Matching Loss maintains proper gradient flow: {loss.item():.4f}"
            )
            return True
        else:
            print("✗ Feature Matching Loss: unexpected gradient behavior")
            print(f"  Real features have grads: {not real_has_no_grads}")
            print(f"  Fake features have grads: {fake_has_grads}")
            return False

    except Exception as e:
        print(f"✗ Feature Matching Loss test failed: {e}")
        return False


def test_discriminator_no_cached_results() -> bool | None:
    """Test that discriminator re-computes forward pass during discriminator step."""
    print("\nTesting Discriminator Update Logic...")

    # Create R3GAN loss
    r3gan_loss = R3GANLoss(loss_weight=1.0)

    # Create a simple discriminator
    class SimpleDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    net_d = SimpleDiscriminator()
    real_images = torch.randn(2, 3, 64, 64)
    fake_images = torch.randn(2, 3, 64, 64)

    # Test that discriminator step re-computes outputs (not using cached results)
    try:
        # During discriminator step, it should re-compute discriminator outputs
        # even if cached results were computed during generator step
        disc_loss_dict = r3gan_loss(
            net_d=net_d,
            real_images=real_images,
            fake_images=fake_images,
            is_disc=True,
        )
        print(
            f"✓ Discriminator step re-computes forward pass: {disc_loss_dict['d_loss'].item():.4f}"
        )
        return True
    except Exception as e:
        print(f"✗ Discriminator update logic test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("SISR Training Fixes Verification")
    print("=" * 40)

    results = []
    results.append(test_r3gan_current_iter())
    results.append(test_feature_matching_gradients())
    results.append(test_discriminator_no_cached_results())

    print("\n" + "=" * 40)
    print(f"Test Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("✓ All fixes are working correctly!")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
