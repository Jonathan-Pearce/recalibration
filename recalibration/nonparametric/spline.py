"""Spline Calibration – natural cubic spline in log-odds space.

Fits a monotonically regularised natural cubic spline to map predicted
probabilities to calibrated probabilities.  Knots are placed at fixed
quantiles of the validation scores and a smoothness penalty (integrated
squared second derivative) is applied.

Reference:
    Gupta, C., Ramdas, A. & Balsubramani, A. (2021).
    "Distribution-Free Calibration Guarantees for Histogram Binning
    without Sample Splitting."  ICML 2021.
    (Natural cubic spline calibration variant.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from recalibration.base import Calibrator


class SplineCalibrator(Calibrator):
    r"""Per-class spline calibration.

    For each class a natural cubic spline is parameterised by learnable
    values at ``n_knots`` fixed quantile knots.  Piecewise-linear
    interpolation between knot values is used at inference time, making
    the calibration map differentiable almost everywhere.

    A smoothness penalty penalises squared differences between adjacent
    knot values, acting as a discrete proxy for the integrated squared
    second derivative.

    Parameters
    ----------
    model : nn.Module
        Pre-trained classifier.
    num_classes : int
        Number of output classes.
    n_knots : int
        Number of interior knots (default 6).
    reg : float
        Smoothness regularisation weight (default 1e-3).
    freeze_model : bool
        Freeze the wrapped model parameters (default *True*).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        n_knots: int = 6,
        reg: float = 1e-3,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(model, freeze_model=freeze_model)
        self.num_classes = num_classes
        self.n_knots = n_knots
        self.reg = reg

        # Knot locations in [0, 1] – fixed, equally spaced including endpoints
        knots = torch.linspace(0.0, 1.0, n_knots + 2)  # includes 0 and 1
        self.register_buffer("knots", knots)

        # Learnable knot *values* – initialised to identity mapping
        self.knot_values = nn.Parameter(
            knots.unsqueeze(0).expand(num_classes, -1).clone()
        )

    def _interpolate(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Piecewise-linear interpolation through knot values for class *class_idx*.

        Parameters
        ----------
        x : 1-D Tensor  (batch,)
            Input probabilities in [0, 1].
        class_idx : int
            Which class's spline to use.

        Returns
        -------
        1-D Tensor  (batch,)
        """
        knots = self.knots  # [K+2]
        values = self.knot_values[class_idx]  # [K+2]

        # Bucket each x into a knot interval
        # searchsorted: finds where x would be inserted
        idx = torch.searchsorted(knots.contiguous(), x.contiguous()).clamp(
            1, knots.size(0) - 1
        )

        # Left and right knot for each x
        x0 = knots[idx - 1]
        x1 = knots[idx]
        y0 = values[idx - 1]
        y1 = values[idx]

        # Linear interpolation fraction
        t = ((x - x0) / (x1 - x0).clamp(min=1e-8)).clamp(0.0, 1.0)
        return y0 + t * (y1 - y0)

    def _smoothness_penalty(self) -> torch.Tensor:
        """Discrete smoothness penalty: sum of squared second differences."""
        # Second differences of knot values
        kv = self.knot_values
        d2 = kv[:, 2:] - 2 * kv[:, 1:-1] + kv[:, :-2]
        return self.reg * d2.pow(2).sum()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits through the per-class spline calibration curves."""
        probs = torch.softmax(logits, dim=-1).clamp(1e-7, 1.0 - 1e-7)
        calibrated = torch.empty_like(probs)

        for k in range(self.num_classes):
            calibrated[:, k] = self._interpolate(probs[:, k], k)

        # Clamp and renormalise
        calibrated = calibrated.clamp(min=1e-8)
        row_sums = calibrated.sum(dim=1, keepdim=True).clamp(min=1e-8)
        calibrated = calibrated / row_sums

        return torch.log(calibrated.clamp(min=1e-8))

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> "SplineCalibrator":
        """Optimise spline knot values with L-BFGS + smoothness penalty.

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
        val_logits = val_logits.detach().to(self.knot_values.device)
        val_labels = val_labels.detach().to(self.knot_values.device)

        optimizer = torch.optim.LBFGS([self.knot_values], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = (
                F.cross_entropy(self.calibrate(val_logits), val_labels)
                + self._smoothness_penalty()
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
