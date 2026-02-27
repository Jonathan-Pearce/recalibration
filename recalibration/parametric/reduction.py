"""OVA and TvA reduction strategies for multi-class calibration via BCE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.base import Calibrator


class _ReductionBase(Calibrator):
    """Shared logic for binary-reduction calibration methods.

    Each class in ``[0, K)`` is calibrated independently via a per-class
    scale and bias learned using binary cross-entropy.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(model, freeze_model=freeze_model)
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply per-class affine scaling."""
        return logits * self.scale + self.bias

    # Subclasses must implement ``_build_targets``.

    def _build_targets(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Return per-class binary target tensor ``[N, C]``."""
        raise NotImplementedError

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> "_ReductionBase":
        """Optimise per-class scale/bias via BCE loss.

        Parameters
        ----------
        val_logits : Tensor ``[N, C]``
        val_labels : Tensor ``[N]``
        max_iter : int
        lr : float

        Returns
        -------
        self
        """
        val_logits = val_logits.detach().to(self.scale.device)
        val_labels = val_labels.detach().to(self.scale.device)
        targets = self._build_targets(val_logits, val_labels)

        optimizer = torch.optim.LBFGS(
            [self.scale, self.bias], lr=lr, max_iter=max_iter
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            cal = self.calibrate(val_logits)
            loss = F.binary_cross_entropy_with_logits(cal, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


class OVAReduction(_ReductionBase):
    """One-versus-All reduction.

    Every class is treated as a separate binary problem: the target for
    class *k* is 1 when the ground-truth label equals *k* and 0 otherwise.
    """

    def _build_targets(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return F.one_hot(labels, self.num_classes).float()


class TvAReduction(_ReductionBase):
    """Top-versus-All reduction.

    Only the *predicted* (top-1) class receives a positive target; all
    other classes are 0.  This focuses calibration effort on the most
    confident class per sample.
    """

    def _build_targets(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels).float()
        targets = torch.zeros_like(logits)
        targets[torch.arange(logits.size(0), device=logits.device), preds] = correct
        return targets
