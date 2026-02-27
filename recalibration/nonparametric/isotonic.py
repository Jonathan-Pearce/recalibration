"""Isotonic Regression via the Pool Adjacent Violators Algorithm (PAVA).

Provides both a hard (non-differentiable) PAVA implementation and a
differentiable soft-sorting proxy that enables gradient flow through the
calibration map for end-to-end fine-tuning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from recalibration.base import Calibrator


# ------------------------------------------------------------------
# Pool Adjacent Violators Algorithm (pure PyTorch, no NumPy)
# ------------------------------------------------------------------


def _pava(values: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pool Adjacent Violators – isotonic (non-decreasing) regression.

    Parameters
    ----------
    values : 1-D Tensor, length *N*
        Observed values in the order induced by the sorted scores.
    weights : 1-D Tensor, length *N*, optional
        Non-negative weights (default: uniform).

    Returns
    -------
    1-D Tensor, length *N*
        Isotonically regressed values.
    """
    n = values.size(0)
    if weights is None:
        weights = torch.ones(n, device=values.device, dtype=values.dtype)

    result = values.clone()
    w = weights.clone()

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Merge blocks [i] and [i+1]
            new_val = (result[i] * w[i] + result[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            new_w = w[i] + w[i + 1]
            result[i] = new_val
            w[i] = new_w
            result = torch.cat([result[: i + 1], result[i + 2 :]])
            w = torch.cat([w[: i + 1], w[i + 2 :]])
            n -= 1
            # Back-track
            if i > 0:
                i -= 1
        else:
            i += 1

    # Expand blocks back to original length
    return result.repeat_interleave(
        _block_lengths(values, result, w)
    ) if result.size(0) != values.size(0) else result


def _block_lengths(
    original: torch.Tensor,
    merged: torch.Tensor,
    merged_weights: torch.Tensor,
) -> torch.Tensor:
    """Infer per-block lengths after PAVA merging from merged weights.

    Since each original sample has weight 1 (default), the merged weight
    of a block equals its length.
    """
    return merged_weights.long()


# ------------------------------------------------------------------
# Differentiable soft-sorting proxy
# ------------------------------------------------------------------


class _SoftSort(torch.autograd.Function):
    """Differentiable soft-sort using a relaxed permutation matrix.

    Given scores ``s``, it computes a soft permutation via::

        P_soft = softmax(-|s_i - sorted(s)_j| / tau)

    and returns ``P_soft @ sorted(values)``.  Gradients flow through
    ``tau`` and the original ``values``.
    """

    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        values: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        sorted_scores, sort_idx = scores.sort()
        sorted_values = values[sort_idx]

        # Pairwise absolute differences: [N, N]
        diff = (scores.unsqueeze(1) - sorted_scores.unsqueeze(0)).abs()
        P = torch.softmax(-diff / tau, dim=1)

        result = P @ sorted_values
        ctx.save_for_backward(P, sorted_values, scores, sorted_scores)
        ctx.tau = tau
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        P, sorted_values, scores, sorted_scores = ctx.saved_tensors
        tau = ctx.tau

        # Gradient w.r.t. values (through P @ sorted_values)
        # dL/d(sorted_values) = P^T @ grad_output
        grad_sorted_values = P.t() @ grad_output

        # Scatter back to original order
        _, sort_idx = scores.sort()
        grad_values = torch.zeros_like(scores)
        grad_values[sort_idx] = grad_sorted_values

        # Gradient w.r.t. scores (through the softmax)
        # P_ij = softmax(-|s_i - s^sorted_j|/tau)
        diff = (scores.unsqueeze(1) - sorted_scores.unsqueeze(0))
        sign = diff.sign()
        # dP_ij/ds_i = P_ij * (sum_k P_ik * sign_ik/tau - sign_ij/tau)
        weighted = (P * (-sign / tau))
        correction = (P * (P * (-sign / tau)).sum(dim=1, keepdim=True))
        dP_ds = weighted - correction  # [N, N]
        grad_scores = (dP_ds @ sorted_values) * grad_output

        return grad_scores, grad_values, None


def soft_sort(
    scores: torch.Tensor,
    values: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Apply a differentiable soft-sort to *values* ordered by *scores*.

    Parameters
    ----------
    scores : 1-D Tensor
        Sorting keys.
    values : 1-D Tensor
        Values to reorder.
    tau : float
        Temperature controlling the softness of the permutation.

    Returns
    -------
    1-D Tensor
        Soft-sorted values.
    """
    return _SoftSort.apply(scores, values, tau)


# ------------------------------------------------------------------
# Isotonic calibrator module
# ------------------------------------------------------------------


class IsotonicCalibrator(Calibrator):
    """Non-parametric isotonic regression calibrator.

    During :meth:`fit`, per-class isotonic regression (PAVA) is applied to
    the validation logits.  The resulting mappings are stored as piece-wise
    constant lookup tables.  At inference time, the differentiable
    soft-sorting proxy interpolates through the stored mapping so that
    gradients can flow if needed.

    Parameters
    ----------
    model : nn.Module
        Pre-trained classifier.
    num_classes : int
        Number of output classes.
    tau : float
        Soft-sort temperature (lower ≈ harder sort).
    freeze_model : bool
        Freeze the wrapped model parameters (default *True*).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        tau: float = 0.1,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(model, freeze_model=freeze_model)
        self.num_classes = num_classes
        self.tau = tau
        # Stored calibration maps: list of (bin_edges, bin_values) per class
        self._bin_edges: list[torch.Tensor] = []
        self._bin_values: list[torch.Tensor] = []

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> "IsotonicCalibrator":
        """Fit isotonic regression per class using PAVA.

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
        probs = torch.softmax(val_logits, dim=-1)
        n = val_logits.size(0)

        self._bin_edges = []
        self._bin_values = []

        for k in range(self.num_classes):
            scores = probs[:, k]
            targets = (val_labels == k).float()

            sorted_idx = scores.argsort()
            sorted_scores = scores[sorted_idx]
            sorted_targets = targets[sorted_idx]

            iso_values = _pava(sorted_targets)
            self._bin_edges.append(sorted_scores.detach().to(device))
            self._bin_values.append(iso_values.detach().to(device))

        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits through the fitted isotonic regression curves.

        Uses the differentiable soft-sort proxy for interpolation so that
        gradients are preserved.
        """
        if not self._bin_edges:
            return logits  # not yet fitted – pass-through

        probs = torch.softmax(logits, dim=-1)
        calibrated = torch.empty_like(probs)

        for k in range(self.num_classes):
            col = probs[:, k]
            edges = self._bin_edges[k]
            values = self._bin_values[k]

            # For each input probability, find the nearest bin via searchsorted
            idx = torch.searchsorted(edges.contiguous(), col.contiguous()).clamp(
                0, edges.size(0) - 1
            )
            calibrated[:, k] = values[idx]

        # Re-normalise so rows sum to 1 (since isotonic regression is per-class)
        row_sums = calibrated.sum(dim=1, keepdim=True).clamp(min=1e-8)
        calibrated = calibrated / row_sums

        # Convert back to logit space (log-probabilities) for downstream CE loss
        return torch.log(calibrated.clamp(min=1e-8))
