import numpy as np
from scipy.integrate import solve_ivp
from random import randint

class LorenzSimulator:
    """
    Simulates different configurations of the Lorenz system, including the standard Lorenz system,
    coupled Lorenz systems, and the Lorenz 96 system. Provides data transformation functionalities
    to standardize or manipulate the simulation outputs.

    Attributes:
        config (dict): Configuration for the simulator including parameters like initial state and system constants.

    Methods:
        simulate_lorenz(): Runs a simulation of the currently configured Lorenz system.
        generate_data_set(): Generates a data set from the chosen Lorenz system by running a simulation.
        split_data(data, train_size=0.8): Splits the generated data into training and testing sets.
    """
    def __init__(self, function_name='lorenz', config=None):
        """
        Initializes the LorenzSimulator with default parameters or a provided configuration.

        Parameters:
            function_name (str): Specifies the Lorenz system variant to simulate.
            config (dict, optional): Allows overriding default simulation parameters.
        """
        default_config = {
            "initial_state": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "params": (10.0, 28.0, 8/3, 0.5),
            "transient_time": 10,
            "total_time": 2500,
            "num_points": 30000,
            "function_name": function_name,
            "lorenz96_params": {
                "initial_state": np.random.rand(36).tolist(),
                "F": 8.0
            }
        }

        if config:
            # Update the default config with the provided config
            self.config = {**default_config, **config}
        else:
            self.config = default_config

        self.states_data = None

    def lorenz(self, t, state, sigma, rho, beta):
        """
        Represents the Lorenz system's set of differential equations.

        Args:
        - t (float): Current time (unused in the equations themselves but required by solve_ivp).
        - state (list or np.ndarray): The current state (x, y, z) of the system.
        - sigma, rho, beta (float): Parameters of the Lorenz system.

        Returns:
        - list: Derivatives of the state variables.
        """
        x, y, z = state[:3]
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    
    def lorenz_coupled(self, t, state, sigma, rho, beta, coupling_strength):
        """
        Represents the coupled Lorenz systems' set of differential equations.

        Args:
        - t (float): Current time (unused in the equations themselves but required by solve_ivp).
        - state (list or np.ndarray): The current state (x, y, z) of the system.
        - sigma, rho, beta (float): Parameters of the Lorenz system.

        Returns:
        - list: Derivatives of the state variables.
        """
        x_h, y_h, z_h = state[:3]
        x, y, z = state[3:]
        return [
            sigma * (y_h - x_h), 
            x_h * (rho - z_h) - y_h, 
            x_h * y_h - beta * z_h, 
            (sigma + x_h * coupling_strength) * (y - x), 
            x * (rho + y_h * coupling_strength - z) - y, 
            x * y - (beta + z_h * coupling_strength) * z
        ]

    def lorenz96(self, t, x, F):
        """
        Represents the Lorenz 96 system's set of differential equations.

        Args:
        - t (float): Current time.
        - x (list or np.ndarray): The current state of the system.
        - F (float): Forcing term.

        Returns:
        - list: Derivatives of the state variables.
        """
        N = len(x)
        dXdt = np.zeros(N)
        for i in range(N):
            dXdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return dXdt

    def generate_data_set(self):
        """
        Generates a data set from the chosen Lorenz system by running a simulation.
        """
        config = self.config
        full_time_span = (0, config["transient_time"] + config["total_time"])
        
        initial_state = config["initial_state"]
        params = config["params"]
        function_name = config["function_name"]
        
        if function_name == 'lorenz_coupled':
            solution = solve_ivp(fun=self.lorenz_coupled, t_span=full_time_span, y0=initial_state, args=params, method='RK45', dense_output=True)
            states_sample = solution.sol(np.linspace(config["transient_time"], config["transient_time"] + config["total_time"], config["num_points"]))
        elif function_name == 'lorenz':
            initial_state = initial_state[:3]  # Use only the first 3 values for standard Lorenz
            params = params[:3]  # Use only the first 3 parameters for standard Lorenz
            solution = solve_ivp(fun=self.lorenz, t_span=full_time_span, y0=initial_state, args=params, method='RK45', dense_output=True)
            states_sample = solution.sol(np.linspace(config["transient_time"], config["transient_time"] + config["total_time"], config["num_points"]))
        elif function_name == 'lorenz96':
            initial_state = self.config["lorenz96_params"]["initial_state"]
            F = self.config["lorenz96_params"]["F"]
            solution = solve_ivp(fun=self.lorenz96, t_span=full_time_span, y0=initial_state, args=(F,), method='RK45', dense_output=True)
            states_sample = solution.sol(np.linspace(config["transient_time"], config["transient_time"] + config["total_time"], config["num_points"]))
        else:
            raise ValueError(f"Unknown function name: {function_name}")
        
        # Adding noise to the states sample
        noise_level = config.get("noise_level", 0.01)  # Adjust noise level as needed
        noise = np.random.normal(scale=noise_level, size=states_sample.shape)
        noisy_states_sample = states_sample + noise

        self.states_data = self.standardize_data(states_sample.T)

    def standardize_data(self, data):
        """
        Standardizes the data by subtracting the mean and dividing by the standard deviation of each variable.

        Args:
        - data (np.ndarray): Data array where each row represents a different time point and columns represent variables.

        Returns:
        - np.ndarray: The standardized data.
        """
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        standardized_data = (data - means) / stds
        return standardized_data

    def simulate_lorenz(self):
        """
        Simulates the chosen Lorenz system with random initial conditions.
        
        Returns:
        - tuple: Four arrays containing X_train, Y_train, X_test, Y_test.
        """
        function_name = self.config["function_name"]
        
        if function_name == 'lorenz_coupled':
            self.config["initial_state"] = np.random.rand(6)  # Random initial conditions for coupled Lorenz
        elif function_name == 'lorenz':
            self.config["initial_state"] = np.random.rand(3)  # Random initial conditions for standard Lorenz
        elif function_name == 'lorenz96':
            self.config["lorenz96_params"]["initial_state"] = np.random.rand(36).tolist()  # Random initial conditions for Lorenz 96
        
        self.generate_data_set()
        return self.split_data()

    def split_data(self, data=None, train_size=0.8):
        """
        Splits the data into training and testing sets based on a specified proportion.

        Args:
            data (np.ndarray, optional): The complete dataset to be split. If None, uses internally stored data.
            train_size (float): The proportion of the dataset to include in the train split. Value must be between 0 and 1.

        Returns:
            tuple: Four arrays containing X_train, Y_train, X_test, Y_test.
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be a float between 0 and 1.")
        
        if data is None:
            data = self.states_data

        # Calculate the index at which to split the data
        split_idx = int(len(data) * train_size)
        
        # Split data into training and test sets
        X_train = data[:split_idx, :]
        Y_train = data[1:split_idx+1, :]  # Shift by 1 to create the Y set
        X_test = data[split_idx:-1, :]
        Y_test = data[split_idx+1:, :]  # Start Y_test just after X_test
        
        return X_train, Y_train, X_test, Y_test