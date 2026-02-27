"""Beta Calibration – maps predicted probabilities through a beta family.

Beta calibration fits the model::

    calibrated_p = 1 / (1 + 1 / exp(a * log(p / (1 - p)) + b))

which is equivalent to logistic regression on the log-odds, enabling
correction of both over- and under-confidence in a principled way.

Reference:
    Kull, M., Silva Filho, T. & Flach, P. (2017).
    "Beta calibration: a well-founded and easily implemented improvement
    on logistic calibration for binary classifiers."  AISTATS 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.base import Calibrator


class BetaCalibrator(Calibrator):
    """Per-class beta calibration.

    For each class *k*, a two-parameter affine map in log-odds space is
    learned::

        logit_cal_k = a_k * logit(p_k) + b_k

    where ``p_k = softmax(logits)_k``.  The calibrated probability is
    ``sigmoid(logit_cal_k)``, which is then renormalised across classes
    and converted back to log-probabilities.

    Parameters
    ----------
    model : nn.Module
        Pre-trained classifier.
    num_classes : int
        Number of output classes.
    freeze_model : bool
        Freeze the wrapped model parameters (default *True*).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(model, freeze_model=freeze_model)
        self.num_classes = num_classes
        self.a = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply beta calibration in log-odds space."""
        probs = torch.softmax(logits, dim=-1).clamp(1e-7, 1.0 - 1e-7)
        log_odds = torch.log(probs / (1.0 - probs))  # logit transform
        calibrated_logits = self.a * log_odds + self.b
        calibrated_probs = torch.sigmoid(calibrated_logits)
        # Renormalise
        row_sums = calibrated_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        calibrated_probs = calibrated_probs / row_sums
        return torch.log(calibrated_probs.clamp(min=1e-8))

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> "BetaCalibrator":
        """Optimise per-class beta calibration parameters with L-BFGS.

        Parameters
        ----------
        val_logits : Tensor ``[N, C]``
            Raw logits from the wrapped model on a validation set.
        val_labels : Tensor ``[N]``
            Ground-truth class labels.
        max_iter : int
            Maximum L-BFGS iterations.
        lr : float
            Learning rate for L-BFGS.

        Returns
        -------
        self
        """
        val_logits = val_logits.detach().to(self.a.device)
        val_labels = val_labels.detach().to(self.a.device)

        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self.calibrate(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
