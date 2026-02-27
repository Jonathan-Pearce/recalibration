# recalibration

A modular, PyTorch-native post-hoc probability recalibration library for deep neural networks.

## Features

- **Base `Calibrator` wrapper** – wraps any `nn.Module` classifier; freezes backbone parameters by default while keeping calibration parameters trainable.
- **Forward-hook context manager** (`calibration_hook`) – non-invasively intercepts logits via `register_forward_hook`; hooks are always removed on exit to prevent GPU memory leaks.
- **TensorDict support** – first-class compatibility with `tensordict.TensorDict` for multi-modal and RL data streams.
- **Parametric methods**
  - *Temperature Scaling* – single global scalar optimised with L-BFGS.
  - *Vector Scaling* – per-class diagonal scale + bias.
  - *Dirichlet Calibration* – full linear map in log-probability space with ODIR regularisation (λ off-diagonal, μ intercept penalties).
  - *OVA / TvA Reduction* – collapse K-class problems into per-class binary targets trained with BCE loss.
  - *Beta Calibration* – per-class logistic regression in log-odds space, correcting both over- and under-confidence.
- **Non-parametric methods**
  - *Isotonic Regression* – Pool Adjacent Violators Algorithm (PAVA) with a differentiable soft-sorting proxy for gradient flow.
  - *Spline Calibration* – piecewise-linear interpolation through learnable quantile knots with smoothness regularisation.
  - *Bayesian Binning into Quantiles (BBQ)* – Bayesian model averaging over multiple equal-frequency binning schemes.
- **Calibration metrics** – Expected Calibration Error (ECE).

## Installation

```bash
pip install -e .            # core (PyTorch only)
pip install -e ".[dev]"     # development (adds pytest, tensordict)
```

## Quick Start

```python
import torch
from recalibration import TemperatureScaling

model = ...  # your pre-trained classifier
ts = TemperatureScaling(model, init_temperature=1.5)

# Fit on a held-out calibration set
ts.fit(val_logits, val_labels)

# Inference – calibrated logits
calibrated = ts(test_inputs)
```

### Using the forward-hook context manager

```python
from recalibration import calibration_hook

with calibration_hook(model, lambda logits: logits / 1.5):
    output = model(inputs)  # logits are scaled inside the context
# Hook is automatically removed here
```

## Running Tests

```bash
pytest tests/ -v
```
