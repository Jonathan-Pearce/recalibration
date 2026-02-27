"""Tests for Beta Calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.parametric.beta import BetaCalibrator


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestBetaCalibrator:
    def test_initial_a_is_ones(self):
        model = _DummyModel()
        bc = BetaCalibrator(model, num_classes=3)
        torch.testing.assert_close(bc.a.data, torch.ones(3))

    def test_initial_b_is_zeros(self):
        model = _DummyModel()
        bc = BetaCalibrator(model, num_classes=3)
        torch.testing.assert_close(bc.b.data, torch.zeros(3))

    def test_output_shape(self):
        model = _DummyModel()
        bc = BetaCalibrator(model, num_classes=3)
        logits = torch.randn(8, 3)
        out = bc.calibrate(logits)
        assert out.shape == logits.shape

    def test_output_is_log_prob(self):
        """Calibrated output should sum to ~1 in probability space."""
        model = _DummyModel()
        bc = BetaCalibrator(model, num_classes=3)
        logits = torch.randn(16, 3)
        log_probs = bc.calibrate(logits)
        probs = torch.exp(log_probs)
        row_sums = probs.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(16), atol=1e-4, rtol=1e-4)

    def test_gradient_flows(self):
        model = _DummyModel()
        bc = BetaCalibrator(model, num_classes=3)
        logits = torch.randn(4, 3)
        out = bc.calibrate(logits)
        loss = out.sum()
        loss.backward()
        assert bc.a.grad is not None
        assert bc.b.grad is not None

    def test_model_frozen(self):
        model = _DummyModel()
        bc = BetaCalibrator(model, freeze_model=True, num_classes=3)
        for p in bc.model.parameters():
            assert not p.requires_grad

    def test_fit_reduces_nll(self):
        """After fitting, NLL on the calibration set should decrease."""
        torch.manual_seed(42)
        model = _DummyModel()
        bc = BetaCalibrator(model, num_classes=3)

        val_logits = torch.randn(200, 3)
        val_labels = torch.randint(0, 3, (200,))

        nll_before = F.cross_entropy(bc.calibrate(val_logits), val_labels).item()
        bc.fit(val_logits, val_labels)
        nll_after = F.cross_entropy(bc.calibrate(val_logits), val_labels).item()

        assert nll_after <= nll_before
