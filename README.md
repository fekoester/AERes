# AERes — Attention-Enhanced Reservoir Computing

[![tests](https://github.com/fekoester/AERes/actions/workflows/tests.yml/badge.svg)](https://github.com/fekoester/AERes/actions/workflows/tests.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2312.16503-b31b1b.svg)](https://arxiv.org/abs/2312.16503)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AERes is a small PyTorch library for experimenting with the **attention-enhanced reservoir** architecture from
[*Attention-Enhanced Reservoir Computing*](https://arxiv.org/abs/2312.16503). A classic reservoir uses a fixed linear readout.
AERes replaces it with an attention readout whose weights change with the current reservoir state.

What's included:

- **`Reservoir`**: a seeded echo state network with an optional leak rate and standardization of its states.
- **Readouts that share one interface** (`train`, `predict`, `evaluate`, `save_model`, `load_model`):
  - `RidgeTrainer`: the classic closed-form ridge readout.
  - `AttentionTrainer`: the linear or nonlinear attention readout from the paper.
  - `NodeTrajectoryAttentionTrainer`: self-attention in which each reservoir node is a token holding its recent trajectory.
- **Closed-loop forecasting** with `Reservoir.forecast(...)` and the **valid prediction time** metric.
- **Benchmark systems**: Lorenz-63, coupled Lorenz, Lorenz-96, Rössler, Mackey-Glass and Kuramoto-Sivashinsky.

## Installation

```bash
pip install git+https://github.com/fekoester/AERes.git
# with the extra packages used by the examples (matplotlib, scikit-learn):
pip install "AERes[examples] @ git+https://github.com/fekoester/AERes.git"
```

AERes needs Python ≥ 3.9 and PyTorch ≥ 2.0. It uses a GPU automatically if one is available.

## Quick start

```python
import torch
from AERes import (DynamicalSystemSimulator, Reservoir, RidgeTrainer, AttentionTrainer,
                   valid_prediction_time)

torch.manual_seed(0)

# 1. Data: one-step-ahead prediction on the Lorenz attractor (Y[t] = X[t+1])
sim = DynamicalSystemSimulator("lorenz", seed=0)
X_train, Y_train, X_test, Y_test = sim.simulate()

# 2. Reservoir: drive it with the training inputs and standardize the node states
reservoir = Reservoir(X_train, number_nodes=50, input_dimension=3, seed=1, standardize=True)

# 3. Readouts: ridge baseline vs. attention
ridge = RidgeTrainer(reservoir.states_stand, Y_train, alpha=1.0)
ridge.train()
attention = AttentionTrainer(reservoir.states_stand, Y_train, layer_type="linear")
attention.train(epochs=100)

# 4. Test: re-use the *training* standardization (calc_standardize=False) to avoid leakage
reservoir.run_given_inputs(X_test, calc_standardize=False)
ridge.evaluate(reservoir.states_stand, Y_test)
attention.evaluate(reservoir.states_stand, Y_test)
```

### Closed-loop forecasting

With `forecast`, the reservoir first follows the real trajectory for a warm-up period. After that, its own predictions become its next inputs:

```python
reservoir.reset_state()
prediction = reservoir.forecast(attention, X_test[:251], steps=500)  # prediction[0] ≈ Y_test[250]

dt = sim.config["total_time"] / (sim.config["num_points"] - 1)
vpt = valid_prediction_time(prediction, Y_test[250:750], dt=dt, lyapunov_exponent=0.906)
print(f"Valid prediction time: {vpt:.2f} Lyapunov times")
```

`forecast` accepts any of the readouts above, any fitted scikit-learn regressor, or a plain function.
For `NodeTrajectoryAttentionTrainer` it keeps track of the rolling window of states.

### Node-trajectory attention

Each reservoir node becomes a token whose features are its last `window_size` states. Self-attention then learns
how the node trajectories interact:

```python
from AERes import NodeTrajectoryAttentionTrainer

reservoir.reset_state()
reservoir.run_given_inputs(X_train)  # training states (the quick start left test states in the reservoir)
trainer = NodeTrajectoryAttentionTrainer(reservoir.states_stand, Y_train, window_size=32,
                                         d_model=64, nhead=4, pooling="mean")
trainer.train(epochs=100)
```

The pooling options are `"mean"`, `"max"`, `"flatten"`, `"attention"` and `"cls"`.

### Benchmark systems

```python
DynamicalSystemSimulator("rossler", seed=0).simulate()
DynamicalSystemSimulator("mackey_glass", config={"mackey_glass_params": {"tau": 30, "beta": 0.2,
                         "gamma": 0.1, "n": 10, "dt": 0.1, "x0": 1.2}}).simulate()
DynamicalSystemSimulator("kuramoto_sivashinsky", seed=0).simulate()   # 64-dimensional
```

| name | dim | notes |
|---|---|---|
| `lorenz` | 3 | Lorenz-63 |
| `lorenz_coupled` | 6 | two Lorenz systems, the first modulates the second |
| `lorenz96` | 36 | forcing `F = 8` |
| `rossler` | 3 | `a = b = 0.2`, `c = 5.7` |
| `mackey_glass` | 1 | delay `τ = 17` (RK4) |
| `kuramoto_sivashinsky` | 64 | `L = 22`, ETDRK4 spectral solver |

All series get small measurement noise (`noise_level`, 0.01 by default) and are standardized. To make a run
reproducible, pass `seed=`. Without it, NumPy's global random state is used, as in earlier versions.

### Leaky reservoirs

```python
Reservoir(X_train, number_nodes=200, input_dimension=3, leakage=0.3)
```

The update rule is `x(t+1) = (1 − a)·x(t) + a·tanh(W x(t) + W_in u(t) + b)`. The default `leakage=1.0`
is the non-leaky update the paper used.

## Examples

| script | what it does |
|---|---|
| `examples/small_example.py` | end-to-end: ridge vs. attention, closed-loop forecast plot |
| `examples/forecasting_benchmark.py` | NRMSE and valid prediction time for every built-in system |
| `examples/compare_reservoir_attention.py` | ridge vs. AER vs. node-trajectory attention across reservoir sizes (writes plots) |
| `examples/node_trajectory_attention_example.py` | minimal node-trajectory attention run |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you use AERes in your research, please cite:

```bibtex
@article{koester2023attention,
  title   = {Attention-Enhanced Reservoir Computing},
  author  = {K{\"o}ster, Felix and Kanno, Kazutaka and Ohkubo, Jun and Uchida, Atsushi},
  journal = {arXiv preprint arXiv:2312.16503},
  year    = {2023}
}
```

## License

MIT (see [LICENSE](LICENSE)).
