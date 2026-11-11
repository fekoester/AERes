import numpy as np


class RidgeTrainer:
    """
    Classic reservoir-computing readout: ridge regression with an unpenalised intercept,
    solved in closed form. It exposes the same interface as the attention trainers
    (``train``, ``predict``, ``evaluate``, ``MSELoss``, ``save_model``, ``load_model``),
    so readouts can be swapped without changing the surrounding code.

    Equivalent to ``sklearn.linear_model.Ridge(alpha=alpha)`` but without the scikit-learn dependency.
    """

    def __init__(self, X, Y, alpha=1.0):
        """
        Args:
        - X (numpy.ndarray): Reservoir states, shape (samples, nodes).
        - Y (numpy.ndarray): Targets, shape (samples, targets) or (samples,).
        - alpha (float): L2 regularisation strength.
        """
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        if len(self.X) != len(self.Y):
            raise ValueError("X and Y must have the same number of samples.")
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def train(self, epochs=None, verbose=False):
        """
        Fits the readout in closed form. ``epochs`` is accepted for interface compatibility and ignored.

        Returns:
            float: Training MSE.
        """
        x_mean = self.X.mean(axis=0)
        y_mean = self.Y.mean(axis=0)
        Xc = self.X - x_mean
        Yc = self.Y - y_mean
        gram = Xc.T @ Xc
        gram[np.diag_indices_from(gram)] += self.alpha
        self.coef_ = np.linalg.solve(gram, Xc.T @ Yc)  # (nodes, targets)
        self.intercept_ = y_mean - x_mean @ self.coef_
        loss = self._mse(self.predict(self.X), self.Y)
        if verbose:
            print(f"Ridge training MSE: {loss:.4f}")
        return loss

    fit = train

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("RidgeTrainer has not been trained yet. Call train() first.")

    def predict(self, input):
        """Predicts outputs for an array of states, shape (samples, nodes) or (nodes,)."""
        self._check_fitted()
        input = np.asarray(input, dtype=np.float64)
        if input.ndim == 1:
            input = input.reshape(1, -1)
        return input @ self.coef_ + self.intercept_

    @staticmethod
    def _mse(prediction, target):
        return float(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))

    def MSELoss(self, prediction, target):
        """Mean squared error between prediction and target."""
        return self._mse(prediction, target)

    def evaluate(self, X, Y, verbose=True):
        """Returns (and optionally prints) the test MSE."""
        Y = np.asarray(Y)
        loss = self._mse(self.predict(X), Y.reshape(-1, 1) if Y.ndim == 1 else Y)
        if verbose:
            print(f"Test Loss: {loss:.4f}")
        return loss

    def parameter_count(self):
        self._check_fitted()
        return self.coef_.size + self.intercept_.size

    def save_model(self, filepath):
        """Saves the fitted weights to an ``.npz`` file."""
        self._check_fitted()
        np.savez(filepath, coef=self.coef_, intercept=self.intercept_, alpha=self.alpha)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads weights saved with ``save_model``."""
        with np.load(filepath) as data:
            self.coef_ = data["coef"]
            self.intercept_ = data["intercept"]
            self.alpha = float(data["alpha"])
        print(f"Model loaded from {filepath}")
