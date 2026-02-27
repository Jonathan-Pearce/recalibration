"""Tests for Temperature Scaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.parametric.temperature import TemperatureScaling


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestTemperatureScaling:
    def test_calibrate_divides_by_temperature(self):
        model = _DummyModel()
        ts = TemperatureScaling(model, init_temperature=2.0)
        logits = torch.randn(8, 3)
        expected = logits / 2.0
        torch.testing.assert_close(ts.calibrate(logits), expected)

    def test_forward_applies_scaling(self):
        model = _DummyModel()
        ts = TemperatureScaling(model, init_temperature=2.0)
        x = torch.randn(8, 4)
        raw = model(x)
        torch.testing.assert_close(ts(x), raw / 2.0)

    def test_temperature_is_learnable(self):
        model = _DummyModel()
        ts = TemperatureScaling(model)
        assert ts.temperature.requires_grad

    def test_model_frozen(self):
        model = _DummyModel()
        ts = TemperatureScaling(model, freeze_model=True)
        for p in ts.model.parameters():
            assert not p.requires_grad

    def test_fit_reduces_nll(self):
        """After fitting, NLL on the calibration set should decrease."""
        torch.manual_seed(42)
        model = _DummyModel()
        ts = TemperatureScaling(model, init_temperature=3.0)

        val_x = torch.randn(200, 4)
        val_labels = torch.randint(0, 3, (200,))
        val_logits = model(val_x).detach()

        nll_before = F.cross_entropy(ts.calibrate(val_logits), val_labels).item()
        ts.fit(val_logits, val_labels)
        nll_after = F.cross_entropy(ts.calibrate(val_logits), val_labels).item()

        assert nll_after <= nll_before

    def test_gradient_flows(self):
        model = _DummyModel()
        ts = TemperatureScaling(model, init_temperature=2.0)
        logits = torch.randn(4, 3)
        out = ts.calibrate(logits)
        loss = out.sum()
        loss.backward()
        assert ts.temperature.grad is not None
