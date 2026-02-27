"""Tests for Vector Scaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.parametric.vector import VectorScaling


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestVectorScaling:
    def test_initial_calibrate_is_identity(self):
        model = _DummyModel()
        vs = VectorScaling(model, num_classes=3)
        logits = torch.randn(8, 3)
        torch.testing.assert_close(vs.calibrate(logits), logits)

    def test_weight_and_bias_shapes(self):
        model = _DummyModel()
        vs = VectorScaling(model, num_classes=5)
        assert vs.weight.shape == (5,)
        assert vs.bias.shape == (5,)

    def test_gradient_flows(self):
        model = _DummyModel()
        vs = VectorScaling(model, num_classes=3)
        logits = torch.randn(4, 3)
        loss = vs.calibrate(logits).sum()
        loss.backward()
        assert vs.weight.grad is not None
        assert vs.bias.grad is not None

    def test_fit_reduces_nll(self):
        torch.manual_seed(42)
        model = _DummyModel()
        vs = VectorScaling(model, num_classes=3)

        val_x = torch.randn(200, 4)
        val_labels = torch.randint(0, 3, (200,))
        val_logits = model(val_x).detach()

        nll_before = F.cross_entropy(vs.calibrate(val_logits), val_labels).item()
        vs.fit(val_logits, val_labels)
        nll_after = F.cross_entropy(vs.calibrate(val_logits), val_labels).item()

        assert nll_after <= nll_before
