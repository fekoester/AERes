import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystemSimulator:
    """
    Generates benchmark time series from chaotic dynamical systems and splits them into one-step-ahead
    prediction datasets (``Y[t] = X[t+1]``).

    Supported systems (``function_name``):

    ===========================  ===========  ===================================================
    name                         dimension    description
    ===========================  ===========  ===================================================
    ``lorenz``                   3            Lorenz-63 attractor
    ``lorenz_coupled``           6            Two Lorenz systems with parameter coupling
    ``lorenz96``                 36           Lorenz-96 with forcing ``F``
    ``rossler``                  3            Rössler attractor
    ``mackey_glass``             1            Mackey-Glass delay differential equation
    ``kuramoto_sivashinsky``     64           Kuramoto-Sivashinsky PDE (spatiotemporal chaos)
    ===========================  ===========  ===================================================

    All outputs have a small additive measurement noise (``noise_level``) and are standardized per
    variable.

    Parameters:
        function_name (str): One of the systems above.
        config (dict, optional): Overrides for the default configuration (shallow merge).
        seed (int, optional): Seed for initial conditions and noise. If None, NumPy's global random
            state is used (so ``np.random.seed`` still controls reproducibility, as in earlier versions).
    """

    SYSTEMS = ("lorenz", "lorenz_coupled", "lorenz96", "rossler", "mackey_glass", "kuramoto_sivashinsky")

    def __init__(self, function_name='lorenz', config=None, seed=None):
        if function_name not in self.SYSTEMS:
            raise ValueError(f"Unknown function name: {function_name}. Choose one of {self.SYSTEMS}.")
        config = dict(config or {})
        seed = config.pop("seed", seed)
        self.rng = np.random.default_rng(seed) if seed is not None else np.random

        default_config = {
            "initial_state": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "params": (10.0, 28.0, 8 / 3, 0.5),
            "transient_time": 10,
            "total_time": 2500,
            "num_points": 30000,
            "noise_level": 0.01,
            "function_name": function_name,
            "lorenz96_params": {
                "initial_state": self.rng.random(36).tolist(),
                "F": 8.0,
            },
            "rossler_params": {"a": 0.2, "b": 0.2, "c": 5.7},
            "mackey_glass_params": {"beta": 0.2, "gamma": 0.1, "n": 10, "tau": 17.0, "dt": 0.1, "x0": 1.2},
            "ks_params": {"L": 22.0, "N": 64, "dt": 0.25, "warmup_time": 100.0},
        }
        self.config = {**default_config, **config}
        self.config["function_name"] = function_name
        self.states_data = None

    # ------------------------------------------------------------------ vector fields

    @staticmethod
    def lorenz(t, state, sigma, rho, beta):
        """Lorenz-63 vector field."""
        x, y, z = state[:3]
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    @staticmethod
    def lorenz_coupled(t, state, sigma, rho, beta, coupling_strength):
        """Two Lorenz systems; the first modulates the parameters of the second."""
        x_h, y_h, z_h = state[:3]
        x, y, z = state[3:]
        return [
            sigma * (y_h - x_h),
            x_h * (rho - z_h) - y_h,
            x_h * y_h - beta * z_h,
            (sigma + x_h * coupling_strength) * (y - x),
            x * (rho + y_h * coupling_strength - z) - y,
            x * y - (beta + z_h * coupling_strength) * z,
        ]

    @staticmethod
    def lorenz96(t, x, F):
        """Lorenz-96 vector field, vectorized: dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F."""
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    @staticmethod
    def rossler(t, state, a, b, c):
        """Rössler vector field."""
        x, y, z = state
        return [-y - z, x + a * y, b + z * (x - c)]

    # ------------------------------------------------------------------ integrators

    def _sample_times(self):
        c = self.config
        return np.linspace(c["transient_time"], c["transient_time"] + c["total_time"], c["num_points"])

    def _solve_ode(self, fun, y0, args):
        c = self.config
        t_span = (0, c["transient_time"] + c["total_time"])
        solution = solve_ivp(fun=fun, t_span=t_span, y0=y0, args=args, method='RK45', dense_output=True)
        return solution.sol(self._sample_times()).T

    def _integrate_mackey_glass(self):
        """RK4 integration of the Mackey-Glass DDE with a constant initial history."""
        p = self.config["mackey_glass_params"]
        dt, tau = p["dt"], p["tau"]
        beta, gamma, n = p["beta"], p["gamma"], p["n"]
        delay = int(round(tau / dt))
        t_end = self.config["transient_time"] + self.config["total_time"]
        steps = int(np.ceil(t_end / dt)) + 1

        x = np.empty(steps + delay)
        x[:delay + 1] = p["x0"]

        def f(x_t, x_delayed):
            return beta * x_delayed / (1.0 + x_delayed ** n) - gamma * x_t

        for i in range(delay, steps + delay - 1):
            xd0, xd1 = x[i - delay], x[i - delay + 1]
            xdm = 0.5 * (xd0 + xd1)
            k1 = f(x[i], xd0)
            k2 = f(x[i] + 0.5 * dt * k1, xdm)
            k3 = f(x[i] + 0.5 * dt * k2, xdm)
            k4 = f(x[i] + dt * k3, xd1)
            x[i + 1] = x[i] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        times = np.arange(steps) * dt
        return np.interp(self._sample_times(), times, x[delay:]).reshape(-1, 1)

    def _integrate_kuramoto_sivashinsky(self, u0):
        """
        ETDRK4 pseudo-spectral integration of u_t = -u u_x - u_xx - u_xxxx on a periodic domain
        (Kassam & Trefethen, 2005).
        """
        p = self.config["ks_params"]
        L, N, dt = p["L"], int(p["N"]), p["dt"]
        k = 2 * np.pi / L * np.arange(N // 2 + 1)
        lin = k ** 2 - k ** 4
        E, E2 = np.exp(dt * lin), np.exp(dt * lin / 2)

        M = 16
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = dt * lin[:, None] + r[None, :]
        Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
        f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
        f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
        g = -0.5j * k
        g[-1] = 0 if N % 2 == 0 else g[-1]

        def nonlinear(v):
            return g * np.fft.rfft(np.fft.irfft(v, n=N) ** 2)

        v = np.fft.rfft(u0)
        warmup_steps = int(round(p["warmup_time"] / dt))
        t_end = self.config["transient_time"] + self.config["total_time"]
        steps = int(np.ceil(t_end / dt)) + 1
        out = np.empty((steps, N))
        for i in range(warmup_steps + steps):
            if i >= warmup_steps:
                out[i - warmup_steps] = np.fft.irfft(v, n=N)
            Nv = nonlinear(v)
            a = E2 * v + Q * Nv
            Na = nonlinear(a)
            b = E2 * v + Q * Na
            Nb = nonlinear(b)
            c = E2 * a + Q * (2 * Nb - Nv)
            Nc = nonlinear(c)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

        times = np.arange(steps) * dt
        sample_times = self._sample_times()
        return np.column_stack([np.interp(sample_times, times, out[:, j]) for j in range(N)])

    # ------------------------------------------------------------------ data generation

    def generate_data_set(self):
        """Integrates the configured system and stores noisy, standardized data in ``states_data``."""
        config = self.config
        name = config["function_name"]
        params = config["params"]

        if name == 'lorenz':
            data = self._solve_ode(self.lorenz, list(config["initial_state"])[:3], tuple(params[:3]))
        elif name == 'lorenz_coupled':
            data = self._solve_ode(self.lorenz_coupled, list(config["initial_state"]), tuple(params))
        elif name == 'lorenz96':
            p = config["lorenz96_params"]
            data = self._solve_ode(self.lorenz96, np.asarray(p["initial_state"], dtype=float), (p["F"],))
        elif name == 'rossler':
            p = config["rossler_params"]
            data = self._solve_ode(self.rossler, list(config["initial_state"])[:3], (p["a"], p["b"], p["c"]))
        elif name == 'mackey_glass':
            data = self._integrate_mackey_glass()
        elif name == 'kuramoto_sivashinsky':
            u0 = config.get("ks_initial_state")
            if u0 is None:
                u0 = 0.1 * (2 * self.rng.random(int(config["ks_params"]["N"])) - 1)
            data = self._integrate_kuramoto_sivashinsky(np.asarray(u0, dtype=float))
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unknown function name: {name}")

        # Drawn as (dim, time) and transposed to keep the random stream identical to earlier versions.
        noise = self.rng.normal(scale=config.get("noise_level", 0.01), size=data.T.shape).T
        self.states_data = self.standardize_data(data + noise)
        return self.states_data

    @staticmethod
    def standardize_data(data):
        """Standardizes each column to zero mean and unit variance."""
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        stds[stds == 0] = 1
        return (data - means) / stds

    def _randomize_initial_state(self):
        name = self.config["function_name"]
        if name == 'lorenz_coupled':
            self.config["initial_state"] = self.rng.random(6)
        elif name in ('lorenz', 'rossler'):
            self.config["initial_state"] = self.rng.random(3)
        elif name == 'lorenz96':
            self.config["lorenz96_params"] = {**self.config["lorenz96_params"],
                                              "initial_state": self.rng.random(36).tolist()}
        elif name == 'mackey_glass':
            self.config["mackey_glass_params"] = {**self.config["mackey_glass_params"],
                                                  "x0": 0.5 + self.rng.random()}

    def simulate(self, train_size=0.8):
        """
        Simulates the configured system from random initial conditions.

        Returns:
            tuple: X_train, Y_train, X_test, Y_test.
        """
        self._randomize_initial_state()
        self.generate_data_set()
        return self.split_data(train_size=train_size)

    def simulate_lorenz(self, train_size=0.8):
        """Alias of :meth:`simulate`, kept for backwards compatibility."""
        return self.simulate(train_size=train_size)

    def split_data(self, data=None, train_size=0.8):
        """
        Splits a series into one-step-ahead train/test sets.

        Returns:
            tuple: X_train, Y_train, X_test, Y_test, where Y is X shifted by one step.
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be a float between 0 and 1.")
        if data is None:
            data = self.states_data
        split_idx = int(len(data) * train_size)
        X_train = data[:split_idx, :]
        Y_train = data[1:split_idx + 1, :]
        X_test = data[split_idx:-1, :]
        Y_test = data[split_idx + 1:, :]
        return X_train, Y_train, X_test, Y_test


class LorenzSimulator(DynamicalSystemSimulator):
    """Backwards-compatible name for :class:`DynamicalSystemSimulator`."""
