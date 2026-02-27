"""Base calibrator module, forward-hook utilities, and context managers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn as nn

try:
    from tensordict import TensorDict
except ImportError:  # tensordict is optional
    TensorDict = None


class Calibrator(nn.Module):
    """Base post-hoc calibrator that wraps a pre-trained model.

    The wrapped model is stored as a frozen submodule (``requires_grad=False``
    by default).  Subclasses override :meth:`calibrate` to transform raw
    logits into calibrated logits.

    Parameters
    ----------
    model : nn.Module
        A pre-trained classifier whose output is a ``[batch, num_classes]``
        logit tensor.
    freeze_model : bool
        If *True* (default), all parameters of ``model`` are frozen so that
        only calibration parameters receive gradients.
    """

    def __init__(self, model: nn.Module, freeze_model: bool = True) -> None:
        super().__init__()
        self.model = model
        if freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Transform raw logits → calibrated logits.

        Subclasses **must** override this method.
        """
        return logits

    def forward(self, x: Any, **kwargs: Any) -> torch.Tensor:
        """Run the wrapped model and apply calibration.

        ``x`` may be a plain :class:`torch.Tensor` or, when *tensordict*
        is installed, a :class:`~tensordict.TensorDict`.  In the latter case
        the model is called with the TensorDict and the output tensor keyed
        ``"logits"`` is calibrated in-place.
        """
        if TensorDict is not None and isinstance(x, TensorDict):
            out = self.model(x, **kwargs)
            if isinstance(out, TensorDict):
                logits = out["logits"]
                out["logits"] = self.calibrate(logits)
                return out
            # model returned a plain tensor even though input was TensorDict
            return self.calibrate(out)

        logits = self.model(x, **kwargs)
        return self.calibrate(logits)


# ----------------------------------------------------------------------
# Forward-hook based calibration
# ----------------------------------------------------------------------


def _make_calibration_hook(
    calibrate_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable:
    """Return a forward-hook function that applies *calibrate_fn* to module output."""

    def hook(
        module: nn.Module,
        input: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return calibrate_fn(output)

    return hook


@contextmanager
def calibration_hook(
    module: nn.Module,
    calibrate_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    target_layer: Optional[nn.Module] = None,
) -> Iterator[None]:
    """Context manager that non-invasively intercepts logits via a forward hook.

    While the context is active a ``register_forward_hook`` is installed on
    *target_layer* (or on *module* itself when *target_layer* is ``None``).
    The hook is **always** removed on exit to prevent GPU memory leaks.

    Parameters
    ----------
    module : nn.Module
        The model (used as default hook target when *target_layer* is ``None``).
    calibrate_fn : callable
        ``(logits: Tensor) -> Tensor`` transformation applied to the layer output.
    target_layer : nn.Module, optional
        Specific sub-module to attach the hook to.  When ``None``, the hook is
        attached to *module* itself.

    Example
    -------
    >>> with calibration_hook(model, lambda t: t / 1.5):
    ...     calibrated = model(inputs)
    """
    target = target_layer if target_layer is not None else module
    handle = target.register_forward_hook(_make_calibration_hook(calibrate_fn))
    try:
        yield
    finally:
        handle.remove()
