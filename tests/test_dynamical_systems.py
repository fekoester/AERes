import numpy as np
import pytest

from AERes import DynamicalSystemSimulator, LorenzSimulator

SMALL = {"transient_time": 5, "total_time": 40, "num_points": 400}
DIMS = {"lorenz": 3, "lorenz_coupled": 6, "lorenz96": 36, "rossler": 3,
        "mackey_glass": 1, "kuramoto_sivashinsky": 64}


@pytest.mark.parametrize("system", DynamicalSystemSimulator.SYSTEMS)
def test_all_systems(system):
    X_train, Y_train, X_test, Y_test = DynamicalSystemSimulator(system, config=SMALL, seed=0).simulate()
    assert X_train.shape == (320, DIMS[system])
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    np.testing.assert_array_equal(X_train[1:], Y_train[:-1])
    np.testing.assert_array_equal(X_test[1:], Y_test[:-1])


def test_seed_reproducible():
    a = DynamicalSystemSimulator("rossler", config=SMALL, seed=3).simulate()[0]
    b = DynamicalSystemSimulator("rossler", config=SMALL, seed=3).simulate()[0]
    c = DynamicalSystemSimulator("rossler", config=SMALL, seed=4).simulate()[0]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_global_seed_still_controls_lorenz_simulator():
    np.random.seed(7)
    a = LorenzSimulator("lorenz", config=SMALL).simulate_lorenz()[0]
    np.random.seed(7)
    b = LorenzSimulator("lorenz", config=SMALL).simulate_lorenz()[0]
    np.testing.assert_array_equal(a, b)


def test_lorenz96_vectorized_matches_loop():
    x = np.random.default_rng(0).normal(size=36)
    N = len(x)
    loop = np.array([(x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + 8.0 for i in range(N)])
    np.testing.assert_allclose(DynamicalSystemSimulator.lorenz96(0, x, 8.0), loop)


def test_unknown_system():
    with pytest.raises(ValueError):
        DynamicalSystemSimulator("nope")
