# Sec. IV-A — Distributed Linear Quadratic Regulation

This experiment reproduces **Fig. 2** of the paper: a finite-horizon LQR
game with `N = 5` scalar agents under a time-varying sparse network
topology. Five curves are produced:

| Curve in plot                 | Paper label (Fig. 2)         | Source            |
| ----------------------------- | ---------------------------- | ----------------- |
| `centralized LQR`             | LQR baseline                 | closed-form Riccati |
| `furieri`                     | subspace constraints         | zeroth-order [Furieri et al. 2020] |
| `model-based ours known graph`| ours (known graph)           | this work         |
| `model-free ours`             | ours                         | this work (policy gradient) |
| `model-based ours`            | *(not in paper)* additional reference curve using the proximity graph but trained by back-propagating through the cost | this work |

The paper's Fig. 2 reports four of these — the extra `model-based ours`
curve is left in for completeness; ignore it when reading the paper.

## Files

- `case_1_linear_system.py` — single script that trains/runs all five
  methods for `T = 1_000_000` iterations and dumps the cost-vs-iteration
  curves as `loss_case_1_*.npy`.
- `case_1_linear_system_plot.py` — produces the figure from the saved
  `.npy` arrays. Reads freshly-trained curves from the current
  working directory if present, otherwise falls back to
  `precomputed_losses/`.
- `functions.py` — `NeuralNetwork`, `PolicyGradientNeuralNetwork`,
  `RandomPositionsProximity`, `SquashedNormal`.
- `precomputed_losses/` — `.npy` files produced by the run that
  generated Fig. 2, included so reviewers can regenerate the figure in
  seconds without retraining.

## Reproduce the figure

```bash
# Fast path: plot from precomputed losses
python case_1_linear_system_plot.py

# Full retrain (CPU, several hours; iteration count hardcoded at T = 1e6)
python case_1_linear_system.py
python case_1_linear_system_plot.py
```
