"""Evaluation metrics for time-series prediction with reservoirs."""
import numpy as np


def _as_2d(array):
    array = np.asarray(array, dtype=np.float64)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def mse(prediction, target):
    """Mean squared error."""
    return float(np.mean((_as_2d(prediction) - _as_2d(target)) ** 2))


def nrmse(prediction, target):
    """Root mean squared error normalised by the standard deviation of the target."""
    target = _as_2d(target)
    std = np.std(target)
    return float(np.sqrt(mse(prediction, target)) / (std if std > 0 else 1.0))


def normalized_error(prediction, target):
    """
    Per-time-step normalised error ``||y_pred(t) - y(t)|| / sqrt(<||y||^2>)``, the quantity commonly
    used to define the valid prediction time of chaotic forecasts (e.g. Pathak et al., 2018).

    Returns:
        np.ndarray: Shape (time,).
    """
    prediction, target = _as_2d(prediction), _as_2d(target)
    n = min(len(prediction), len(target))
    prediction, target = prediction[:n], target[:n]
    scale = np.sqrt(np.mean(np.sum(target ** 2, axis=1)))
    return np.linalg.norm(prediction - target, axis=1) / (scale if scale > 0 else 1.0)


def valid_prediction_time(prediction, target, threshold=0.4, dt=1.0, lyapunov_exponent=None):
    """
    Valid prediction time (VPT) of an autonomous forecast: the time until the normalised error first
    exceeds ``threshold``.

    Args:
        prediction (np.ndarray): Forecast, shape (time, dim).
        target (np.ndarray): Ground truth, shape (time, dim).
        threshold (float): Error threshold (0.4 is a common choice).
        dt (float): Time between samples.
        lyapunov_exponent (float, optional): If given, the VPT is returned in Lyapunov times
            (``steps * dt * lyapunov_exponent``). For the classic Lorenz-63 system this is ~0.906.

    Returns:
        float: VPT in steps * dt (or Lyapunov times). If the error never exceeds the threshold, the full
        forecast length is returned.
    """
    error = normalized_error(prediction, target)
    exceeded = np.flatnonzero(error > threshold)
    steps = int(exceeded[0]) if exceeded.size else len(error)
    vpt = steps * dt
    if lyapunov_exponent is not None:
        vpt *= lyapunov_exponent
    return float(vpt)
