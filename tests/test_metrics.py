"""Tests for calibration metrics."""

import torch

from recalibration.metrics import expected_calibration_error


class TestECE:
    def test_perfect_calibration(self):
        """A perfectly calibrated model has ECE ≈ 0."""
        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        labels = torch.tensor([0, 1])
        ece = expected_calibration_error(probs, labels, n_bins=10)
        # Both predictions correct with high confidence
        assert ece.item() < 0.2

    def test_ece_is_non_negative(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(100, 5), dim=1)
        labels = torch.randint(0, 5, (100,))
        ece = expected_calibration_error(probs, labels)
        assert ece.item() >= 0.0

    def test_ece_bounded_by_one(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(100, 5), dim=1)
        labels = torch.randint(0, 5, (100,))
        ece = expected_calibration_error(probs, labels)
        assert ece.item() <= 1.0

    def test_overconfident_model_has_high_ece(self):
        """Model that always predicts 100% confidence but is wrong half the time."""
        probs = torch.tensor([[1.0, 0.0]] * 100)
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        ece = expected_calibration_error(probs, labels, n_bins=10)
        # 100% confidence, 50% accuracy → ECE ≈ 0.5
        assert ece.item() > 0.3
