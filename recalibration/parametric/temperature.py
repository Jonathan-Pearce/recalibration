"""Temperature Scaling – single global scalar calibration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.base import Calibrator


class TemperatureScaling(Calibrator):
    """Post-hoc temperature scaling.

    A single learnable scalar ``temperature`` divides the logit vector
    before softmax.  The parameter is optimised with L-BFGS on an NLL
    loss over a held-out calibration set.

    Parameters
    ----------
    model : nn.Module
        Pre-trained classifier.
    init_temperature : float
        Initial temperature value (default 1.5).
    freeze_model : bool
        Freeze the wrapped model parameters (default *True*).
    """

    def __init__(
        self,
        model: nn.Module,
        init_temperature: float = 1.5,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(model, freeze_model=freeze_model)
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature."""
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> "TemperatureScaling":
        """Optimise temperature on a calibration set using L-BFGS.

        Parameters
        ----------
        val_logits : Tensor ``[N, C]``
            Raw logits produced by the wrapped model on a validation set.
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
        val_logits = val_logits.detach().to(self.temperature.device)
        val_labels = val_labels.detach().to(self.temperature.device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self.calibrate(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
