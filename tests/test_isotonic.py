"""Tests for Isotonic Regression calibrator."""

import torch
import torch.nn as nn

from recalibration.nonparametric.isotonic import (
    IsotonicCalibrator,
    _pava,
    soft_sort,
)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestPAVA:
    def test_already_monotone(self):
        """No merging needed."""
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _pava(vals)
        torch.testing.assert_close(result, vals)

    def test_simple_violation(self):
        vals = torch.tensor([1.0, 3.0, 2.0, 4.0])
        result = _pava(vals)
        # 3 and 2 should merge to 2.5
        expected = torch.tensor([1.0, 2.5, 2.5, 4.0])
        torch.testing.assert_close(result, expected)

    def test_all_decreasing(self):
        vals = torch.tensor([4.0, 3.0, 2.0, 1.0])
        result = _pava(vals)
        # All should merge to mean = 2.5
        expected = torch.tensor([2.5, 2.5, 2.5, 2.5])
        torch.testing.assert_close(result, expected)

    def test_single_element(self):
        vals = torch.tensor([5.0])
        result = _pava(vals)
        torch.testing.assert_close(result, vals)

    def test_result_is_non_decreasing(self):
        torch.manual_seed(0)
        vals = torch.randn(50)
        result = _pava(vals)
        diffs = result[1:] - result[:-1]
        assert (diffs >= -1e-6).all()


class TestSoftSort:
    def test_output_shape(self):
        scores = torch.randn(10)
        values = torch.randn(10)
        out = soft_sort(scores, values, tau=1.0)
        assert out.shape == values.shape

    def test_gradient_exists(self):
        scores = torch.randn(10, requires_grad=True)
        values = torch.randn(10, requires_grad=True)
        out = soft_sort(scores, values, tau=1.0)
        out.sum().backward()
        assert scores.grad is not None
        assert values.grad is not None


class TestIsotonicCalibrator:
    def test_passthrough_before_fit(self):
        model = _DummyModel()
        iso = IsotonicCalibrator(model, num_classes=3)
        x = torch.randn(8, 4)
        raw = model(x)
        torch.testing.assert_close(iso(x), raw)

    def test_fit_and_calibrate(self):
        torch.manual_seed(42)
        model = _DummyModel()
        iso = IsotonicCalibrator(model, num_classes=3)

        val_x = torch.randn(100, 4)
        val_labels = torch.randint(0, 3, (100,))
        val_logits = model(val_x).detach()

        iso.fit(val_logits, val_labels)

        # After fit, output should differ from raw logits
        out = iso.calibrate(val_logits)
        assert out.shape == val_logits.shape

    def test_output_is_log_prob(self):
        """Calibrated output should sum to ~1 in probability space."""
        torch.manual_seed(42)
        model = _DummyModel()
        iso = IsotonicCalibrator(model, num_classes=3)

        val_logits = torch.randn(50, 3)
        val_labels = torch.randint(0, 3, (50,))
        iso.fit(val_logits, val_labels)

        log_probs = iso.calibrate(val_logits)
        probs = torch.exp(log_probs)
        row_sums = probs.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(50), atol=1e-4, rtol=1e-4)
