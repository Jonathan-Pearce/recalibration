"""Recalibration – PyTorch-native post-hoc probability recalibration."""

from recalibration.base import Calibrator, calibration_hook
from recalibration.parametric.temperature import TemperatureScaling
from recalibration.parametric.vector import VectorScaling
from recalibration.parametric.dirichlet import DirichletCalibrator
from recalibration.parametric.reduction import OVAReduction, TvAReduction
from recalibration.nonparametric.isotonic import IsotonicCalibrator
from recalibration.metrics import expected_calibration_error

__all__ = [
    "Calibrator",
    "calibration_hook",
    "TemperatureScaling",
    "VectorScaling",
    "DirichletCalibrator",
    "OVAReduction",
    "TvAReduction",
    "IsotonicCalibrator",
    "expected_calibration_error",
]
