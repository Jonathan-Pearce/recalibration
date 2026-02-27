"""Recalibration – PyTorch-native post-hoc probability recalibration."""

from recalibration.base import Calibrator, calibration_hook
from recalibration.metrics import expected_calibration_error
from recalibration.nonparametric.bbq import BBQCalibrator
from recalibration.nonparametric.isotonic import IsotonicCalibrator
from recalibration.nonparametric.spline import SplineCalibrator
from recalibration.parametric.beta import BetaCalibrator
from recalibration.parametric.dirichlet import DirichletCalibrator
from recalibration.parametric.reduction import OVAReduction, TvAReduction
from recalibration.parametric.temperature import TemperatureScaling
from recalibration.parametric.vector import VectorScaling

__all__ = [
    "Calibrator",
    "calibration_hook",
    "TemperatureScaling",
    "VectorScaling",
    "DirichletCalibrator",
    "OVAReduction",
    "TvAReduction",
    "BetaCalibrator",
    "IsotonicCalibrator",
    "SplineCalibrator",
    "BBQCalibrator",
    "expected_calibration_error",
]
