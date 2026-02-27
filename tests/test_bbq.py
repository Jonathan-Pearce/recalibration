"""Tests for Bayesian Binning into Quantiles (BBQ)."""

import torch
import torch.nn as nn

from recalibration.nonparametric.bbq import BBQCalibrator, _equal_freq_bin_probs


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


class TestEqualFreqBinProbs:
    def test_bin_edges_count(self):
        scores = torch.rand(50)
        targets = torch.randint(0, 2, (50,)).float()
        edges, values, ll = _equal_freq_bin_probs(scores, targets, n_bins=5)
        assert edges.size(0) == 6  # n_bins + 1
        assert values.size(0) == 5

    def test_bin_values_in_range(self):
        scores = torch.rand(100)
        targets = torch.randint(0, 2, (100,)).float()
        _, values, _ = _equal_freq_bin_probs(scores, targets, n_bins=10)
        assert (values >= 0.0).all()
        assert (values <= 1.0).all()


class TestBBQCalibrator:
    def test_passthrough_before_fit(self):
        model = _DummyModel()
        bbq = BBQCalibrator(model, num_classes=3)
        logits = torch.randn(8, 3)
        torch.testing.assert_close(bbq.calibrate(logits), logits)

    def test_fit_and_calibrate(self):
        torch.manual_seed(42)
        model = _DummyModel()
        bbq = BBQCalibrator(model, num_classes=3)

        val_logits = torch.randn(100, 3)
        val_labels = torch.randint(0, 3, (100,))

        bbq.fit(val_logits, val_labels)

        out = bbq.calibrate(val_logits)
        assert out.shape == val_logits.shape

    def test_output_is_log_prob(self):
        """Calibrated output should sum to ~1 in probability space."""
        torch.manual_seed(42)
        model = _DummyModel()
        bbq = BBQCalibrator(model, num_classes=3)

        val_logits = torch.randn(50, 3)
        val_labels = torch.randint(0, 3, (50,))
        bbq.fit(val_logits, val_labels)

        log_probs = bbq.calibrate(val_logits)
        probs = torch.exp(log_probs)
        row_sums = probs.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(50), atol=1e-4, rtol=1e-4)

    def test_custom_bin_counts(self):
        model = _DummyModel()
        bbq = BBQCalibrator(model, num_classes=3, bin_counts=[3, 7])
        assert bbq.bin_counts == [3, 7]

    def test_weights_sum_to_one(self):
        """Model weights per class should sum to 1."""
        torch.manual_seed(42)
        model = _DummyModel()
        bbq = BBQCalibrator(model, num_classes=3)

        val_logits = torch.randn(100, 3)
        val_labels = torch.randint(0, 3, (100,))
        bbq.fit(val_logits, val_labels)

        for k in range(3):
            total_weight = sum(w for w, _, _ in bbq._models[k])
            assert abs(total_weight - 1.0) < 1e-5
