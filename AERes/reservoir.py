import warnings
from collections import deque

import numpy as np
import scipy.linalg


class Reservoir:
    """
    A reservoir computing system: a fixed, random recurrent network of tanh nodes whose states are
    read out by a trainable model (ridge regression, linear attention, node-trajectory attention, ...).

    The state update is a leaky-integrator echo state network::

        x(t+1) = (1 - a) * x(t) + a * tanh(W x(t) + W_in u(t) + b)

    where ``a`` is the ``leakage`` rate. ``leakage=1.0`` (the default) is the non-leaky update used
    in the original AERes code and paper.

    Attributes:
        input_dimension (int): The dimensionality of the input data.
        number_nodes (int): The number of neurons in the reservoir.
        leakage (float): The leak rate ``a`` in (0, 1]. 1.0 means no leaky integration.
        spectral_radius (float): The scaling factor for the adjacency matrix's largest eigenvalue.
        node_degree (float): The connectivity degree for the adjacency matrix.
        node_input_degree (float): The connectivity degree for the input weight matrix.
        seed (int): The random seed for reproducibility; uses a random seed if negative.
        standardize (bool): Whether to standardize the output states of the reservoir.
        states (np.ndarray): Raw states from the last ``run_given_inputs`` call, shape (time, nodes).
        states_stand (np.ndarray): Standardized ``states`` (only if ``standardize=True``).
        state (np.ndarray): The current state, shape (nodes,).
        state_stand (np.ndarray): The current standardized state (only if standardization is fitted).
    """

    def __init__(self, inputs=None, number_nodes=100, spectral_radius=0.95, leakage=1.0, node_degree=0.2,
                 input_dimension=1, node_input_degree=0.5, seed=-1, base_path=None, standardize=False):
        """
        Parameters:
            inputs (np.ndarray, optional): Initial inputs, shape (time, input_dimension). If given, the
                reservoir is run on them immediately (and standardization is fitted on them).
            number_nodes (int): The number of neurons in the reservoir.
            spectral_radius (float): The scaling factor for the adjacency matrix's largest eigenvalue.
            leakage (float): Leak rate in (0, 1]. 1.0 (default) reproduces the original non-leaky update.
            node_degree (float): The connectivity degree for the adjacency matrix.
            input_dimension (int): The dimensionality of the input data.
            node_input_degree (float): The connectivity degree for the input weight matrix.
            seed (int): The random seed for reproducibility; uses a random seed if negative.
            base_path: Deprecated and ignored.
            standardize (bool): Whether to standardize the output states of the reservoir.
        """
        if not 0 < leakage <= 1:
            raise ValueError("leakage must be in the interval (0, 1].")
        if base_path:
            warnings.warn("base_path is unused and will be removed in a future release.",
                          DeprecationWarning, stacklevel=2)

        self.seed = seed if seed >= 0 else np.random.randint(100000)
        self.rng = np.random.default_rng(self.seed)

        self.input_dimension = input_dimension
        self.number_nodes = number_nodes
        self.leakage = leakage
        self.spectral_radius = spectral_radius
        self.node_degree = node_degree
        self.node_input_degree = node_input_degree
        self.standardize = standardize

        self.biases = self.rng.normal(0, 1.0, self.number_nodes)
        self.state = np.zeros(self.number_nodes, dtype=np.float64)
        self.states = np.zeros((0, self.number_nodes), dtype=np.float64)

        self._initialize_matrices()
        if inputs is not None:
            self.run_given_inputs(inputs)

    # ------------------------------------------------------------------ construction

    def _initialize_matrices(self):
        self._construct_adjacency_matrix()
        self._construct_input_matrix()

    def _construct_adjacency_matrix(self):
        """Builds the sparse random recurrent weight matrix and rescales it to ``spectral_radius``."""
        connect_mask = self.rng.uniform(size=(self.number_nodes, self.number_nodes)) < self.node_degree
        scale = np.sqrt(1.0 / (self.number_nodes * self.node_degree))
        self.adjacency_weight_matrix = np.where(
            connect_mask, self.rng.normal(0, scale, (self.number_nodes, self.number_nodes)), 0)

        max_eigenvalue = np.max(np.abs(scipy.linalg.eigvals(self.adjacency_weight_matrix)))
        if max_eigenvalue > 0:
            self.adjacency_weight_matrix *= self.spectral_radius / max_eigenvalue

    def _construct_input_matrix(self):
        """Builds the sparse random input weight matrix."""
        connect_mask = self.rng.uniform(size=(self.number_nodes, self.input_dimension)) < self.node_input_degree
        scale = np.sqrt(1.0 / (self.input_dimension * self.node_input_degree))
        self.input_weights = np.where(
            connect_mask, self.rng.normal(0, scale, (self.number_nodes, self.input_dimension)), 0)

    # ------------------------------------------------------------------ dynamics

    def _update(self, input_term):
        activation = np.tanh(self.adjacency_weight_matrix @ self.state + input_term + self.biases)
        if self.leakage == 1.0:
            self.state = activation
        else:
            self.state = (1.0 - self.leakage) * self.state + self.leakage * activation
        return self.state

    def reset_state(self, state=None):
        """Resets the current reservoir state to zeros (or to the given state)."""
        self.state = np.zeros(self.number_nodes) if state is None else np.array(state, dtype=np.float64)

    def one_step(self, inputs, standardize=False):
        """
        Advances the reservoir by one time step.

        Parameters:
            inputs (np.ndarray): Input vector for the current time step, shape (input_dimension,).
            standardize (bool): If True, also updates ``self.state_stand`` using the fitted
                standardization parameters. This is O(nodes); the state history is not touched.

        Returns:
            np.ndarray: The updated (raw) state of the reservoir.
        """
        state = self._update(self.input_weights @ np.asarray(inputs, dtype=np.float64))
        if standardize:
            self.state_stand = self.standardize_state(state)
        return state

    def run_given_inputs(self, inputs, calc_standardize=True):
        """
        Drives the reservoir with a sequence of inputs, storing every state in ``self.states``.

        Parameters:
            inputs (np.ndarray): Shape (time, input_dimension).
            calc_standardize (bool): If True (and ``standardize`` is on), fit the standardization on
                these states. Use False for test data so the training statistics are reused.
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        # Project all inputs at once; only the recurrence has to stay sequential.
        input_terms = inputs @ self.input_weights.T

        self.states = np.empty((len(inputs), self.number_nodes))
        for i, input_term in enumerate(input_terms):
            self.states[i] = self._update(input_term)

        if self.standardize:
            if calc_standardize:
                self.calculate_standardize_and_apply()
            else:
                self.reapply_standardize()
        return self.states

    # ------------------------------------------------------------------ standardization

    def calculate_standardize_and_apply(self):
        """Fits per-node mean/std on ``self.states`` and applies them."""
        if self.states.size == 0:
            raise ValueError("No states available to standardize.")
        self.col_means = np.mean(self.states, axis=0)
        self.col_stds = np.std(self.states, axis=0)
        self.col_stds[self.col_stds == 0] = 1
        self.reapply_standardize()

    def _check_standardization(self):
        if not hasattr(self, 'col_means') or not hasattr(self, 'col_stds'):
            raise ValueError("Standardization parameters are not available. Please calculate them first.")

    def standardize_state(self, state):
        """Applies the fitted standardization to a single state or an array of states."""
        self._check_standardization()
        return (state - self.col_means) / self.col_stds

    def reapply_standardize(self):
        """Reapplies previously computed standardization parameters to ``states`` and ``state``."""
        self._check_standardization()
        if self.states.size == 0:
            raise ValueError("No states available to reapply standardization.")
        self.states_stand = self.standardize_state(self.states)
        self.state_stand = self.standardize_state(self.state)

    # ------------------------------------------------------------------ forecasting

    def forecast(self, readout, warmup_inputs, steps, output_transform=None):
        """
        Closed-loop (autonomous) forecasting.

        The reservoir is first driven with ``warmup_inputs`` (teacher forcing) to synchronise it with
        the trajectory. Then the readout's prediction is fed back as the next input for ``steps`` steps.

        ``prediction[0]`` is the readout's estimate of the value that follows ``warmup_inputs[-1]``.

        Parameters:
            readout: Any object with a ``predict`` method (``AttentionTrainer``, ``RidgeTrainer``,
                ``NodeTrajectoryAttentionTrainer``, a fitted scikit-learn regressor, ...) or a callable
                mapping a state batch to predictions.
            warmup_inputs (np.ndarray): Shape (warmup_steps, input_dimension).
            steps (int): Number of autonomous steps to predict.
            output_transform (callable, optional): Maps a readout prediction to the next reservoir
                input (useful if the readout predicts something other than the next input).

        Returns:
            np.ndarray: Predictions of shape (steps, output_dimension).
        """
        warmup_inputs = np.asarray(warmup_inputs, dtype=np.float64)
        if warmup_inputs.ndim == 1:
            warmup_inputs = warmup_inputs.reshape(-1, 1)
        predict = readout.predict if hasattr(readout, "predict") else readout
        window_size = getattr(readout, "window_size", None)
        if window_size is not None and len(warmup_inputs) < window_size:
            raise ValueError("warmup_inputs must contain at least window_size steps.")

        use_stand = self.standardize
        if use_stand:
            self._check_standardization()
        history = deque(maxlen=window_size or 1)

        def current_features():
            history.append(self.standardize_state(self.state) if use_stand else self.state.copy())
            if window_size is None:
                return history[-1].reshape(1, -1)
            return np.stack(history, axis=1)[None]  # (1, nodes, window_size)

        for u in warmup_inputs[:-1]:
            self.one_step(u)
            if window_size is not None:
                current_features()

        next_input = warmup_inputs[-1]
        predictions = []
        for _ in range(steps):
            self.one_step(next_input)
            prediction = np.asarray(predict(current_features())).reshape(-1)
            predictions.append(prediction)
            next_input = output_transform(prediction) if output_transform else prediction
        if use_stand:
            self.state_stand = self.standardize_state(self.state)
        return np.asarray(predictions)
