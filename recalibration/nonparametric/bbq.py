"""Bayesian Binning into Quantiles (BBQ).

BBQ is a non-parametric calibration method that considers multiple
equal-frequency binning schemes simultaneously and combines their
predictions using Bayesian model averaging.

Reference:
    Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015).
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning."
    AAAI 2015.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from recalibration.base import Calibrator

# ------------------------------------------------------------------
# Single equal-frequency binning model
# ------------------------------------------------------------------


def _equal_freq_bin_probs(
    scores: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Fit an equal-frequency (quantile) binning model.

    Parameters
    ----------
    scores : 1-D Tensor, length *N*
        Predicted probabilities for a single class.
    targets : 1-D Tensor, length *N*
        Binary targets (1 if ground-truth is this class, else 0).
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    bin_edges : 1-D Tensor, length ``n_bins + 1``
        Boundaries of the quantile bins.
    bin_values : 1-D Tensor, length ``n_bins``
        Average target value within each bin.
    log_likelihood : float
        Log-likelihood of the binning model.
    """
    n = scores.size(0)
    sorted_idx = scores.argsort()
    sorted_targets = targets[sorted_idx]
    sorted_scores = scores[sorted_idx]

    bin_edges = [sorted_scores[0].item()]
    bin_values = []
    log_lik = 0.0

    for i in range(n_bins):
        start = (i * n) // n_bins
        end = ((i + 1) * n) // n_bins
        if end <= start:
            end = start + 1  # ensure at least one sample

        bin_t = sorted_targets[start:end]
        p = bin_t.mean().item()
        p = max(min(p, 1.0 - 1e-7), 1e-7)  # numerical safety
        bin_values.append(p)

        # Bernoulli log-likelihood
        pos = bin_t.sum().item()
        neg = bin_t.size(0) - pos
        log_lik += pos * torch.log(torch.tensor(p)).item()
        log_lik += neg * torch.log(torch.tensor(1.0 - p)).item()

        if i < n_bins - 1:
            # Edge between this bin and next
            edge_idx = min(end, n - 1)
            bin_edges.append(sorted_scores[edge_idx].item())

    bin_edges.append(sorted_scores[-1].item() + 1e-6)

    device = scores.device
    return (
        torch.tensor(bin_edges, device=device),
        torch.tensor(bin_values, device=device),
        log_lik,
    )


def _predict_binned(
    scores: torch.Tensor,
    bin_edges: torch.Tensor,
    bin_values: torch.Tensor,
) -> torch.Tensor:
    """Look up calibrated probabilities from a binning model."""
    idx = (
        torch.searchsorted(bin_edges.contiguous(), scores.contiguous()).clamp(
            1, bin_edges.size(0) - 1
        )
        - 1
    )
    idx = idx.clamp(0, bin_values.size(0) - 1)
    return bin_values[idx]


# ------------------------------------------------------------------
# BBQ calibrator
# ------------------------------------------------------------------


class BBQCalibrator(Calibrator):
    """Bayesian Binning into Quantiles (BBQ) calibrator.

    Considers multiple equal-frequency binning schemes (with different
    numbers of bins) and weights them by their marginal likelihood
    (approximated by the Bernoulli log-likelihood plus a BIC-like
    penalty).

    Parameters
    ----------
    model : nn.Module
        Pre-trained classifier.
    num_classes : int
        Number of output classes.
    bin_counts : list of int, optional
        Candidate numbers of bins to consider (default ``[2, 5, 10, 15, 20]``).
    freeze_model : bool
        Freeze the wrapped model parameters (default *True*).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        bin_counts: List[int] | None = None,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(model, freeze_model=freeze_model)
        self.num_classes = num_classes
        self.bin_counts = bin_counts or [2, 5, 10, 15, 20]

        # Fitted state: per-class list of (weight, bin_edges, bin_values)
        self._models: list[list[Tuple[float, torch.Tensor, torch.Tensor]]] = []

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> "BBQCalibrator":
        """Fit BBQ calibration per class on a validation set.

        Parameters
        ----------
        val_logits : Tensor ``[N, C]``
            Raw logits from the wrapped model on a validation set.
        val_labels : Tensor ``[N]``
            Ground-truth class labels.

        Returns
        -------
        self
        """
        device = val_logits.device
        probs = torch.softmax(val_logits.detach(), dim=-1)
        n = val_logits.size(0)

        self._models = []

        for k in range(self.num_classes):
            scores = probs[:, k]
            targets = (val_labels == k).float().to(device)

            class_models: list[Tuple[float, torch.Tensor, torch.Tensor]] = []
            log_scores: list[float] = []

            for b in self.bin_counts:
                if b >= n:
                    continue
                edges, values, log_lik = _equal_freq_bin_probs(scores, targets, b)
                # BIC-like penalty: -0.5 * b * log(n)
                score = log_lik - 0.5 * b * torch.log(torch.tensor(float(n))).item()
                log_scores.append(score)
                class_models.append((score, edges.detach(), values.detach()))

            if not class_models:
                # Fallback: single bin with the overall mean
                mean_t = targets.mean().clamp(1e-7, 1.0 - 1e-7)
                edges = torch.tensor([0.0, 1.0 + 1e-6], device=device)
                values = torch.tensor([mean_t.item()], device=device)
                self._models.append([(1.0, edges, values)])
                continue

            # Normalise scores into weights via softmax
            log_scores_t = torch.tensor(log_scores)
            weights = torch.softmax(log_scores_t, dim=0)

            weighted_models = [
                (weights[i].item(), class_models[i][1], class_models[i][2])
                for i in range(len(class_models))
            ]
            self._models.append(weighted_models)

        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits through the BBQ calibration ensemble.

        Returns calibrated log-probabilities.
        """
        if not self._models:
            return logits  # not yet fitted – pass-through

        probs = torch.softmax(logits, dim=-1)
        calibrated = torch.zeros_like(probs)

        for k in range(self.num_classes):
            col = probs[:, k]
            for weight, edges, values in self._models[k]:
                calibrated[:, k] += weight * _predict_binned(col, edges, values)

        # Renormalise
        row_sums = calibrated.sum(dim=1, keepdim=True).clamp(min=1e-8)
        calibrated = calibrated / row_sums

        return torch.log(calibrated.clamp(min=1e-8))
