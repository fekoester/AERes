import numpy as np
import pytest

from AERes import RidgeTrainer, mse, nrmse, valid_prediction_time


def test_ridge_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(200, 12)), rng.normal(size=(200, 2))
    ours = RidgeTrainer(X, Y, alpha=0.5)
    ours.train()
    ref = sklearn.Ridge(alpha=0.5).fit(X, Y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-10)


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    W = rng.normal(size=(4, 2))
    ridge = RidgeTrainer(X, X @ W + 3.0, alpha=1e-8)
    assert ridge.train() < 1e-12
    np.testing.assert_allclose(ridge.coef_, W, atol=1e-6)
    np.testing.assert_allclose(ridge.intercept_, 3.0, atol=1e-6)


def test_ridge_save_load(tmp_path):
    rng = np.random.default_rng(2)
    X, Y = rng.normal(size=(50, 3)), rng.normal(size=50)
    a = RidgeTrainer(X, Y)
    a.train()
    path = tmp_path / "ridge.npz"
    a.save_model(path)
    b = RidgeTrainer(X, Y)
    b.load_model(path)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_ridge_requires_training():
    with pytest.raises(RuntimeError):
        RidgeTrainer(np.zeros((3, 2)), np.zeros(3)).predict(np.zeros((1, 2)))


def test_metrics():
    target = np.array([[1.0], [2.0], [3.0]])
    assert mse(target, target) == 0
    assert nrmse(target + 1, target) == pytest.approx(1 / np.std(target))


def test_valid_prediction_time():
    target = np.ones((10, 2))
    prediction = target.copy()
    prediction[6:] += 5
    assert valid_prediction_time(prediction, target) == 6
    assert valid_prediction_time(prediction, target, dt=0.5, lyapunov_exponent=2.0) == 6.0
    assert valid_prediction_time(target, target) == 10
