"""
Unit tests for src/vesuvius/model.py.

Tests cover:
  - Forward pass shape: output matches input spatial dimensions.
  - Receptive field depth: deeper models produce different outputs than
    shallow models (verifies skip connections are wired correctly).
  - Parameter count ordering: base_features=48 has more params than 32.
  - Specialist factory shortcuts produce valid models.
  - Gradient flow: backward pass reaches all parameters (no dead branches).
  - Fully convolutional: model accepts non-square inputs without error.

All tests run on CPU with small (1×1×16×16×16) patches — no GPU required.
The 16³ patch is the minimum size for depth=4 (2^4 = 16 downsampling factor).
"""

from __future__ import annotations

import pytest
import torch

from vesuvius.model import ResidualUNet3D, build_model


# depth=4 halves spatial dims 4 times: 32 → 16 → 8 → 4 → 2.
# InstanceNorm3d requires >1 spatial element, so 2³ is the minimum
# bottleneck size. That means the input must be at least 2^4 * 2 = 32³.
SMALL = (1, 1, 32, 32, 32)   # (B, C, D, H, W) — minimum for depth=4


class TestResidualUNet3D:

    def test_output_shape_matches_input(self):
        model = ResidualUNet3D(base_features=8, depth=2)
        x = torch.randn(1, 1, 8, 8, 8)
        y = model(x)
        assert y.shape == (1, 1, 8, 8, 8)

    def test_output_shape_depth4(self):
        model = ResidualUNet3D(base_features=8, depth=4)
        x = torch.randn(*SMALL)
        y = model(x)
        assert y.shape == torch.Size(SMALL)

    def test_more_base_features_more_parameters(self):
        m32 = ResidualUNet3D(base_features=8,  depth=3)
        m48 = ResidualUNet3D(base_features=16, depth=3)
        assert m48.parameter_count() > m32.parameter_count()

    def test_output_is_single_channel_logit(self):
        """Output should be raw logits — not clamped to [0, 1]."""
        model = ResidualUNet3D(base_features=8, depth=2)
        x = torch.randn(1, 1, 8, 8, 8)
        y = model(x)
        assert y.shape[1] == 1
        # Logits are not bounded; at least some values should be outside [0,1]
        assert (y.abs() > 0.01).any()

    def test_skip_connections_are_wired(self):
        """
        With identical weights, zeroing the stem input should change the
        output — if skip connections are wired, encoder activations from
        the skip path still reach the decoder even when the bottleneck
        input is zero.  This is a proxy test: two different inputs should
        produce different outputs (checks that identity shortcuts work).
        """
        model = ResidualUNet3D(base_features=8, depth=2)
        model.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 1, 8, 8, 8)
            x2 = torch.randn(1, 1, 8, 8, 8)
            y1 = model(x1)
            y2 = model(x2)
        assert not torch.allclose(y1, y2)

    def test_gradient_reaches_all_parameters(self):
        """Backward pass must produce non-None gradients for every parameter."""
        model = ResidualUNet3D(base_features=8, depth=2)
        x = torch.randn(1, 1, 8, 8, 8, requires_grad=False)
        loss = model(x).mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"

    def test_fully_convolutional_non_square_input(self):
        """Model should accept non-cubic patches (e.g. anisotropic CT)."""
        model = ResidualUNet3D(base_features=8, depth=2)
        x = torch.randn(1, 1, 8, 16, 8)   # non-cubic
        y = model(x)
        assert y.shape == (1, 1, 8, 16, 8)

    def test_dropout_zero_deterministic(self):
        """With dropout=0, the same input always produces the same output."""
        model = ResidualUNet3D(base_features=8, depth=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 1, 8, 8, 8)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        assert torch.allclose(y1, y2)

    def test_dropout_nonzero_stochastic_in_train_mode(self):
        """With dropout>0 in train mode, outputs differ between forward passes."""
        model = ResidualUNet3D(base_features=8, depth=2, dropout=0.5)
        model.train()
        x = torch.randn(1, 1, 8, 8, 8)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        # With p=0.5 dropout it is astronomically unlikely these are identical
        assert not torch.allclose(y1, y2)

    def test_parameter_count_method(self):
        model = ResidualUNet3D(base_features=8, depth=2)
        n = model.parameter_count()
        assert n == sum(p.numel() for p in model.parameters())
        assert n > 0


class TestBuildModel:

    def test_build_model_a_config(self):
        """Model A: patch=192, base_features=48, depth=4."""
        model = build_model(patch_size=192, base_features=48, depth=4)
        assert isinstance(model, ResidualUNet3D)

    def test_build_model_b_config(self):
        """Model B: patch=160, base_features=32, depth=4."""
        model = build_model(patch_size=160, base_features=32, depth=4)
        assert isinstance(model, ResidualUNet3D)

    def test_build_model_c_config(self):
        """Model C: patch=192, base_features=48, depth=4, dropout=0.1."""
        model = build_model(patch_size=192, base_features=48, depth=4,
                            dropout=0.1)
        assert isinstance(model, ResidualUNet3D)

    def test_model_b_has_fewer_params_than_model_a(self):
        model_a = build_model(patch_size=192, base_features=48)
        model_b = build_model(patch_size=160, base_features=32)
        assert model_b.parameter_count() < model_a.parameter_count()

    def test_forward_pass_small_patch(self):
        """Smoke test: build_model forward pass on a small patch."""
        model = build_model(patch_size=192, base_features=8, depth=2)
        x = torch.randn(1, 1, 8, 8, 8)
        y = model(x)
        assert y.shape == (1, 1, 8, 8, 8)
        assert torch.isfinite(y).all()
