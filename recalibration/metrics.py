"""Calibration metrics – Expected Calibration Error and helpers."""

from __future__ import annotations

import torch


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Compute the Expected Calibration Error (ECE).

    Parameters
    ----------
    probs : Tensor, shape ``[N, C]``
        Predicted probabilities (after softmax).
    labels : Tensor, shape ``[N]``
        Ground-truth class indices.
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    Tensor
        Scalar ECE value.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device, dtype=probs.dtype)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        mask = (confidences.gt(bin_boundaries[i])) & (
            confidences.le(bin_boundaries[i + 1])
        )
        count = mask.float().sum()
        if count > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].float().mean()
            ece += (avg_conf - avg_acc).abs() * (count / probs.size(0))

    return ece.squeeze()
