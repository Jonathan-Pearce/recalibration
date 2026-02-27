"""Tests for OVA and TvA reduction strategies."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.parametric.reduction import OVAReduction, TvAReduction


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestOVAReduction:
    def test_targets_are_one_hot(self):
        model = _DummyModel()
        ova = OVAReduction(model, num_classes=3)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        targets = ova._build_targets(logits, labels)
        expected = F.one_hot(labels, 3).float()
        torch.testing.assert_close(targets, expected)

    def test_initial_calibrate_identity(self):
        model = _DummyModel()
        ova = OVAReduction(model, num_classes=3)
        logits = torch.randn(4, 3)
        torch.testing.assert_close(ova.calibrate(logits), logits)

    def test_gradient_flows(self):
        model = _DummyModel()
        ova = OVAReduction(model, num_classes=3)
        logits = torch.randn(4, 3)
        loss = ova.calibrate(logits).sum()
        loss.backward()
        assert ova.scale.grad is not None
        assert ova.bias.grad is not None

    def test_fit_reduces_bce(self):
        torch.manual_seed(42)
        model = _DummyModel()
        ova = OVAReduction(model, num_classes=3)
        val_logits = torch.randn(200, 3)
        val_labels = torch.randint(0, 3, (200,))

        targets = ova._build_targets(val_logits, val_labels)
        bce_before = F.binary_cross_entropy_with_logits(
            ova.calibrate(val_logits), targets
        ).item()

        ova.fit(val_logits, val_labels)
        bce_after = F.binary_cross_entropy_with_logits(
            ova.calibrate(val_logits), targets
        ).item()

        assert bce_after <= bce_before


class TestTvAReduction:
    def test_targets_only_top_class(self):
        model = _DummyModel()
        tva = TvAReduction(model, num_classes=3)
        logits = torch.tensor([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0]])
        labels = torch.tensor([0, 2])  # first correct, second wrong

        targets = tva._build_targets(logits, labels)

        # First sample: predicted=0, label=0 → correct → target[0,0]=1
        assert targets[0, 0].item() == 1.0
        assert targets[0, 1].item() == 0.0

        # Second sample: predicted=1, label=2 → wrong → target[1,1]=0
        assert targets[1, 1].item() == 0.0

    def test_gradient_flows(self):
        model = _DummyModel()
        tva = TvAReduction(model, num_classes=3)
        logits = torch.randn(4, 3)
        loss = tva.calibrate(logits).sum()
        loss.backward()
        assert tva.scale.grad is not None
