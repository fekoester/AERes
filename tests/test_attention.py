import numpy as np
import pytest

torch = pytest.importorskip("torch")

from AERes import (AttentionTrainer, LinearAttention, NodeTrajectoryAttentionTrainer, Reservoir,
                   create_node_trajectory_windows)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 12)).astype(np.float32), rng.normal(size=(200, 3)).astype(np.float32)


def test_linear_attention_default_device():
    model = LinearAttention(4, 2)  # used to crash with device=None
    assert model(torch.zeros(5, 4)).shape == (5, 2)


def test_windows_match_naive_loop():
    states = np.random.default_rng(1).normal(size=(20, 5))
    naive = np.array([states[i - 3:i + 1].T for i in range(3, 20)], dtype=np.float32)
    np.testing.assert_array_equal(create_node_trajectory_windows(states, 4), naive)


@pytest.mark.parametrize("layer_type", ["linear", "nonlinear"])
def test_attention_trainer_learns(data, layer_type):
    torch.manual_seed(0)
    X, Y = data
    trainer = AttentionTrainer(X, Y, use_gpu=False, layer_type=layer_type, learning_rate=0.01)
    losses = trainer.train(epochs=15, verbose=False)
    assert losses[-1] < losses[0]
    assert trainer.predict(X).shape == (200, 3)
    assert trainer.predict(X[0]).shape == (1, 3)
    assert isinstance(trainer.evaluate(X, Y, verbose=False), float)


def test_unknown_layer_type(data):
    with pytest.raises(ValueError):
        AttentionTrainer(*data, layer_type="quadratic")


def test_save_load(tmp_path, data):
    X, Y = data
    a = AttentionTrainer(X, Y, use_gpu=False)
    a.train(epochs=1, verbose=False)
    a.save_model(tmp_path / "m.pt")
    b = AttentionTrainer(X, Y, use_gpu=False)
    b.load_model(tmp_path / "m.pt")
    np.testing.assert_allclose(a.predict(X), b.predict(X))


@pytest.mark.parametrize("pooling", ["mean", "max", "flatten", "attention", "cls"])
def test_node_trajectory_trainer(data, pooling):
    X, Y = data
    trainer = NodeTrajectoryAttentionTrainer(X, Y, window_size=4, d_model=8, nhead=2,
                                             pooling=pooling, use_gpu=False)
    trainer.train(epochs=1, verbose=False)
    assert trainer.predict(X).shape == (197, 3)
    assert trainer.evaluate(X, Y, verbose=False) > 0


def test_forecast_with_node_trajectory_readout(data):
    X, Y = data
    reservoir = Reservoir(X[:, :3], number_nodes=12, input_dimension=3, seed=0, standardize=True)
    trainer = NodeTrajectoryAttentionTrainer(reservoir.states_stand, X[:, :3], window_size=5,
                                             d_model=8, nhead=2, use_gpu=False)
    reservoir.reset_state()
    assert reservoir.forecast(trainer, X[:10, :3], steps=7).shape == (7, 3)
