"""Parametric calibration methods."""

from recalibration.parametric.temperature import TemperatureScaling
from recalibration.parametric.vector import VectorScaling
from recalibration.parametric.dirichlet import DirichletCalibrator
from recalibration.parametric.reduction import OVAReduction, TvAReduction

__all__ = [
    "TemperatureScaling",
    "VectorScaling",
    "DirichletCalibrator",
    "OVAReduction",
    "TvAReduction",
]
