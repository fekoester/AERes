import numpy as np
import pytest

from AERes import Reservoir


@pytest.fixture
def inputs():
    return np.random.default_rng(0).normal(size=(300, 3))


def test_seed_reproducible(inputs):
    a = Reservoir(inputs, number_nodes=20, input_dimension=3, seed=5)
    b = Reservoir(inputs, number_nodes=20, input_dimension=3, seed=5)
    np.testing.assert_array_equal(a.states, b.states)


def test_spectral_radius(inputs):
    r = Reservoir(inputs, number_nodes=30, input_dimension=3, seed=1, spectral_radius=0.8)
    assert np.max(np.abs(np.linalg.eigvals(r.adjacency_weight_matrix))) == pytest.approx(0.8)


def test_default_leakage_matches_plain_tanh_update(inputs):
    r = Reservoir(None, number_nodes=10, input_dimension=3, seed=2)
    state = np.zeros(10)
    for u in inputs[:20]:
        state = np.tanh(r.adjacency_weight_matrix @ state + r.input_weights @ u + r.biases)
        r.one_step(u)
    np.testing.assert_allclose(r.state, state)


def test_leakage_is_applied(inputs):
    full = Reservoir(None, number_nodes=10, input_dimension=3, seed=2, leakage=1.0)
    leaky = Reservoir(None, number_nodes=10, input_dimension=3, seed=2, leakage=0.3)
    activation = np.tanh(full.input_weights @ inputs[0] + full.biases)
    np.testing.assert_allclose(leaky.one_step(inputs[0]), 0.3 * activation)


@pytest.mark.parametrize("leakage", [0.0, 1.5])
def test_invalid_leakage(leakage):
    with pytest.raises(ValueError):
        Reservoir(None, number_nodes=5, leakage=leakage)


def test_run_matches_one_step(inputs):
    r = Reservoir(inputs, number_nodes=15, input_dimension=3, seed=3)
    r2 = Reservoir(None, number_nodes=15, input_dimension=3, seed=3)
    stepped = np.array([r2.one_step(u).copy() for u in inputs])
    np.testing.assert_allclose(r.states, stepped)


def test_standardization_reused_on_test_data(inputs):
    r = Reservoir(inputs, number_nodes=15, input_dimension=3, seed=3, standardize=True)
    np.testing.assert_allclose(r.states_stand.mean(axis=0), 0, atol=1e-10)
    means = r.col_means.copy()
    r.run_given_inputs(inputs[:50] * 3, calc_standardize=False)
    np.testing.assert_array_equal(r.col_means, means)
    np.testing.assert_allclose(r.states_stand, (r.states - means) / r.col_stds)


def test_one_step_standardize_updates_state_stand(inputs):
    r = Reservoir(inputs, number_nodes=15, input_dimension=3, seed=3, standardize=True)
    r.one_step(inputs[0], standardize=True)
    np.testing.assert_allclose(r.state_stand, (r.state - r.col_means) / r.col_stds)


def test_forecast_with_identity_readout_feeds_back(inputs):
    r = Reservoir(None, number_nodes=10, input_dimension=3, seed=4)
    calls = []

    def readout(features):
        calls.append(features.shape)
        return np.ones((1, 3))

    out = r.forecast(readout, inputs[:5], steps=4)
    assert out.shape == (4, 3)
    assert calls == [(1, 10)] * 4
    np.testing.assert_array_equal(out, 1.0)
