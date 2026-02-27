"""Vector Scaling – per-class diagonal transformation and bias."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.base import Calibrator


class VectorScaling(Calibrator):
    """Post-hoc vector scaling (diagonal + bias).

    Each class receives its own scale factor and bias term::

        calibrated_logits = diag(w) @ logits + b

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
        self.weight = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply diagonal scaling and bias."""
        return logits * self.weight + self.bias

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> "VectorScaling":
        """Optimise scaling parameters with L-BFGS.

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
        val_logits = val_logits.detach().to(self.weight.device)
        val_labels = val_labels.detach().to(self.weight.device)

        optimizer = torch.optim.LBFGS(
            [self.weight, self.bias], lr=lr, max_iter=max_iter
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self.calibrate(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
