"""Tests that all public symbols are importable from the top-level package."""

import recalibration


class TestPublicAPI:
    def test_all_exports_importable(self):
        for name in recalibration.__all__:
            assert hasattr(recalibration, name), f"{name} listed in __all__ but not importable"

    def test_all_expected_exports(self):
        expected = {
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
        }
        assert set(recalibration.__all__) == expected
