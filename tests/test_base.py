"""Tests for the base Calibrator, forward-hook context manager, and TensorDict."""

import torch
import torch.nn as nn
import pytest

from recalibration.base import Calibrator, calibration_hook


# ── helpers ────────────────────────────────────────────────────────

class _DummyModel(nn.Module):
    """Simple linear classifier for testing."""

    def __init__(self, in_features: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _DoubleCalibrator(Calibrator):
    """Trivial subclass that doubles logits."""

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * 2


# ── Base Calibrator ───────────────────────────────────────────────


class TestCalibrator:
    def test_forward_passthrough(self):
        model = _DummyModel()
        cal = Calibrator(model)
        x = torch.randn(8, 4)
        # Base calibrator is identity – output should match model
        torch.testing.assert_close(cal(x), model(x))

    def test_freeze_model(self):
        model = _DummyModel()
        cal = Calibrator(model, freeze_model=True)
        for p in cal.model.parameters():
            assert not p.requires_grad

    def test_unfreeze_model(self):
        model = _DummyModel()
        cal = Calibrator(model, freeze_model=False)
        for p in cal.model.parameters():
            assert p.requires_grad

    def test_subclass_calibrate(self):
        model = _DummyModel()
        cal = _DoubleCalibrator(model)
        x = torch.randn(4, 4)
        expected = model(x) * 2
        torch.testing.assert_close(cal(x), expected)

    def test_device_migration(self):
        """nn.Parameters should move with .to(device)."""
        model = _DummyModel()
        cal = _DoubleCalibrator(model)
        cal = cal.to("cpu")  # trivial but confirms no error
        x = torch.randn(2, 4)
        out = cal(x)
        assert out.device.type == "cpu"


# ── Forward-hook context manager ─────────────────────────────────


class TestCalibrationHook:
    def test_hook_applies_transform(self):
        model = _DummyModel()
        x = torch.randn(4, 4)
        baseline = model(x)

        with calibration_hook(model, lambda t: t / 2.0):
            hooked = model(x)

        torch.testing.assert_close(hooked, baseline / 2.0)

    def test_hook_removed_after_exit(self):
        model = _DummyModel()
        x = torch.randn(4, 4)
        baseline = model(x)

        with calibration_hook(model, lambda t: t * 0):
            pass  # hook active only inside

        after = model(x)
        torch.testing.assert_close(after, baseline)

    def test_hook_removed_on_exception(self):
        model = _DummyModel()
        x = torch.randn(4, 4)

        with pytest.raises(RuntimeError):
            with calibration_hook(model, lambda t: t):
                raise RuntimeError("boom")

        # Hook should be cleaned up despite exception
        baseline = model(x)
        after = model(x)
        torch.testing.assert_close(after, baseline)

    def test_target_layer_hook(self):
        model = _DummyModel()
        x = torch.randn(4, 4)

        with calibration_hook(model, lambda t: t + 100, target_layer=model.fc):
            out = model(x)

        expected = model.fc(x) + 100
        torch.testing.assert_close(out, expected)


# ── TensorDict compatibility ────────────────────────────────────


class TestTensorDictCompat:
    def test_tensordict_forward(self):
        tensordict = pytest.importorskip("tensordict")
        TensorDict = tensordict.TensorDict

        class TDModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, td):
                return TensorDict({"logits": self.fc(td["features"])})

        model = TDModel()
        cal = _DoubleCalibrator(model)

        td = TensorDict({"features": torch.randn(4, 4)})
        out = cal(td)

        assert isinstance(out, TensorDict)
        expected = model.fc(td["features"]) * 2
        torch.testing.assert_close(out["logits"], expected)
