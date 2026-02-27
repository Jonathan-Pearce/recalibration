"""Dirichlet Calibration with Off-Diagonal and Intercept Regularisation (ODIR)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.base import Calibrator


class DirichletCalibrator(Calibrator):
    r"""Dirichlet calibration with ODIR regularisation.

    Applies a full linear map in log-probability space::

        calibrated = W @ log_softmax(logits) + b

    with regularisation penalties on the off-diagonal entries of ``W``
    (controlled by :math:`\lambda`) and on the intercept ``b`` (controlled
    by :math:`\mu`).

    Parameters
    ----------
    model : nn.Module
        Pre-trained classifier.
    num_classes : int
        Number of output classes.
    lam : float
        Off-diagonal regularisation strength (:math:`\lambda`).
    mu : float
        Intercept regularisation strength (:math:`\mu`).
    freeze_model : bool
        Freeze the wrapped model parameters (default *True*).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lam: float = 1e-3,
        mu: float = 1e-3,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(model, freeze_model=freeze_model)
        self.num_classes = num_classes
        self.lam = lam
        self.mu = mu
        self.W = nn.Parameter(torch.eye(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the Dirichlet linear map in log-probability space."""
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs @ self.W.t() + self.b

    def _odir_penalty(self) -> torch.Tensor:
        """Compute the ODIR regularisation term."""
        off_diag_mask = ~torch.eye(
            self.num_classes, dtype=torch.bool, device=self.W.device
        )
        off_diag = self.W[off_diag_mask]
        return self.lam * off_diag.pow(2).sum() + self.mu * self.b.pow(2).sum()

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> "DirichletCalibrator":
        """Optimise W and b with L-BFGS + ODIR regularisation.

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
        val_logits = val_logits.detach().to(self.W.device)
        val_labels = val_labels.detach().to(self.W.device)

        optimizer = torch.optim.LBFGS([self.W, self.b], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            cal = self.calibrate(val_logits)
            loss = F.cross_entropy(cal, val_labels) + self._odir_penalty()
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
