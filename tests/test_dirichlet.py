"""Tests for Dirichlet Calibration with ODIR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.parametric.dirichlet import DirichletCalibrator


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestDirichletCalibrator:
    def test_initial_W_is_identity(self):
        model = _DummyModel()
        dc = DirichletCalibrator(model, num_classes=3)
        torch.testing.assert_close(dc.W.data, torch.eye(3))

    def test_odir_penalty_zero_at_init(self):
        """Off-diagonal of identity is 0, bias is 0 → penalty = 0."""
        model = _DummyModel()
        dc = DirichletCalibrator(model, num_classes=3, lam=1.0, mu=1.0)
        assert dc._odir_penalty().item() == 0.0

    def test_odir_penalty_increases_with_off_diag(self):
        model = _DummyModel()
        dc = DirichletCalibrator(model, num_classes=3, lam=1.0, mu=0.0)
        dc.W.data[0, 1] = 5.0  # large off-diagonal
        assert dc._odir_penalty().item() > 0.0

    def test_gradient_flows(self):
        model = _DummyModel()
        dc = DirichletCalibrator(model, num_classes=3)
        logits = torch.randn(4, 3)
        loss = dc.calibrate(logits).sum()
        loss.backward()
        assert dc.W.grad is not None
        assert dc.b.grad is not None

    def test_fit_reduces_loss(self):
        torch.manual_seed(42)
        model = _DummyModel()
        dc = DirichletCalibrator(model, num_classes=3)

        val_x = torch.randn(200, 4)
        val_labels = torch.randint(0, 3, (200,))
        val_logits = model(val_x).detach()

        def total_loss(cal):
            return (
                F.cross_entropy(cal.calibrate(val_logits), val_labels)
                + cal._odir_penalty()
            ).item()

        loss_before = total_loss(dc)
        dc.fit(val_logits, val_labels)
        loss_after = total_loss(dc)

        assert loss_after <= loss_before
