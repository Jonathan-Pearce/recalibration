"""Recalibration – PyTorch-native post-hoc probability recalibration."""

from recalibration.base import Calibrator, calibration_hook
from recalibration.parametric.temperature import TemperatureScaling
from recalibration.parametric.vector import VectorScaling
from recalibration.parametric.dirichlet import DirichletCalibrator
from recalibration.parametric.reduction import OVAReduction, TvAReduction
from recalibration.parametric.beta import BetaCalibrator
from recalibration.nonparametric.isotonic import IsotonicCalibrator
from recalibration.nonparametric.spline import SplineCalibrator
from recalibration.nonparametric.bbq import BBQCalibrator
from recalibration.metrics import expected_calibration_error

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
