"""Tests for Spline Calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.nonparametric.spline import SplineCalibrator


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestSplineCalibrator:
    def test_knot_values_init_to_identity(self):
        model = _DummyModel()
        sc = SplineCalibrator(model, num_classes=3, n_knots=6)
        expected = torch.linspace(0.0, 1.0, 8).unsqueeze(0).expand(3, -1)
        torch.testing.assert_close(sc.knot_values.data, expected)

    def test_output_shape(self):
        model = _DummyModel()
        sc = SplineCalibrator(model, num_classes=3)
        logits = torch.randn(8, 3)
        out = sc.calibrate(logits)
        assert out.shape == logits.shape

    def test_output_is_log_prob(self):
        """Calibrated output should sum to ~1 in probability space."""
        model = _DummyModel()
        sc = SplineCalibrator(model, num_classes=3)
        logits = torch.randn(16, 3)
        log_probs = sc.calibrate(logits)
        probs = torch.exp(log_probs)
        row_sums = probs.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(16), atol=1e-4, rtol=1e-4)

    def test_smoothness_penalty_zero_at_init(self):
        """With identity-mapped knots (linear), second differences should be ~0."""
        model = _DummyModel()
        sc = SplineCalibrator(model, num_classes=3, n_knots=6, reg=1.0)
        penalty = sc._smoothness_penalty()
        assert penalty.item() < 1e-10

    def test_gradient_flows(self):
        model = _DummyModel()
        sc = SplineCalibrator(model, num_classes=3)
        logits = torch.randn(4, 3)
        out = sc.calibrate(logits)
        loss = out.sum()
        loss.backward()
        assert sc.knot_values.grad is not None

    def test_model_frozen(self):
        model = _DummyModel()
        sc = SplineCalibrator(model, freeze_model=True, num_classes=3)
        for p in sc.model.parameters():
            assert not p.requires_grad

    def test_fit_reduces_nll(self):
        """After fitting, NLL on the calibration set should decrease."""
        torch.manual_seed(42)
        model = _DummyModel()
        sc = SplineCalibrator(model, num_classes=3, n_knots=4)

        val_logits = torch.randn(200, 3)
        val_labels = torch.randint(0, 3, (200,))

        nll_before = F.cross_entropy(sc.calibrate(val_logits), val_labels).item()
        sc.fit(val_logits, val_labels)
        nll_after = F.cross_entropy(sc.calibrate(val_logits), val_labels).item()

        assert nll_after <= nll_before
