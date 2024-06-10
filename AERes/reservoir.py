import numpy as np
import scipy.linalg

class Reservoir:
    """
    A class for implementing a reservoir computing system, utilizing a fixed, random recurrent neural
    network (the reservoir) and a trainable output layer. This model is particularly suited for tasks
    where capturing complex temporal dynamics is crucial.

    Attributes:
        input_dimension (int): The dimensionality of the input data.
        number_nodes (int): The number of neurons in the reservoir.
        leakage (float): The leakage rate of the reservoir's state update.
        spectral_radius (float): The scaling factor for the adjacency matrix's largest eigenvalue.
        node_degree (float): The connectivity degree for the adjacency matrix.
        node_input_degree (float): The connectivity degree for the input weight matrix.
        seed (int): The random seed for reproducibility; uses a random seed if negative.
        base_path (str): The base directory for caching matrices to avoid repeated computations.
        standardize (bool): Whether to standardize the output states of the reservoir.

    Methods:
        run_given_inputs(inputs, calc_standardize=True): Processes multiple inputs through the reservoir.
        calculate_standardize_and_apply(): Standardizes the state sequences of the reservoir.
        reapply_standardize(): Reapplies previously computed standardization parameters.
    """

    def __init__(self, inputs, number_nodes=100, spectral_radius=0.95, leakage=0.15, node_degree=0.2,
                 input_dimension=1, node_input_degree=0.5, seed=-1, base_path='', standardize=False):
        """
        Initializes a new Reservoir instance with the specified parameters and seeds the random number generator.

        Parameters:
            inputs (np.ndarray): Initial set of inputs to feed into the reservoir.
            number_nodes (int): The number of neurons in the reservoir.
            spectral_radius (float): The scaling factor for the adjacency matrix's largest eigenvalue.
            leakage (float): The leakage rate of the reservoir's state update.
            node_degree (float): The connectivity degree for the adjacency matrix.
            input_dimension (int): The dimensionality of the input data.
            node_input_degree (float): The connectivity degree for the input weight matrix.
            seed (int): The random seed for reproducibility; uses a random seed if negative.
            base_path (str): The base directory for caching matrices; avoids repeated computations.
            standardize (bool): Whether to standardize the output states of the reservoir.
        """
        self.seed = seed if seed >= 0 else np.random.randint(100000)
        self.rng = np.random.default_rng(self.seed)

        self.input_dimension = input_dimension
        self.number_nodes = number_nodes
        self.leakage = leakage
        self.spectral_radius = spectral_radius
        self.node_degree = node_degree
        self.node_input_degree = node_input_degree

        self.standardize = standardize
        self.base_path = base_path

        self.biases = self.rng.normal(0, 1.0, self.number_nodes)
        self.state = np.zeros(self.number_nodes, dtype=np.float64)
        self.states = np.zeros((0, self.number_nodes), dtype=np.float64)

        self._initialize_matrices()
        self.run_given_inputs(inputs)


    def _initialize_matrices(self):
        """
        Private method to construct the adjacency and input weight matrices for the reservoir. This method
        initializes the internal state of the reservoir necessary for subsequent computations.
        """
        self._construct_adjacency_matrix()
        self._construct_input_matrix()

    def _construct_adjacency_matrix(self):
        """
        Private method to construct the adjacency matrix for the reservoir. The adjacency matrix governs the 
        internal connectivity of the reservoir nodes, influencing the dynamics of the system.
        """
        connect_mask = self.rng.uniform(size=(self.number_nodes, self.number_nodes)) < self.node_degree
        scale = np.sqrt(1.0 / (self.number_nodes * self.node_degree))
        self.adjacency_weight_matrix = np.where(connect_mask, self.rng.normal(0, scale, (self.number_nodes, self.number_nodes)), 0)

        eigenvalues = scipy.linalg.eigvals(self.adjacency_weight_matrix)
        self.adjacency_weight_matrix *= self.spectral_radius / np.max(np.abs(eigenvalues))

    def _construct_input_matrix(self):
        """
        Private method to construct the input weight matrix for the reservoir. The input weight matrix
        determines how external inputs are fed into the reservoir nodes.
        """
        connect_mask = self.rng.uniform(size=(self.number_nodes, self.input_dimension)) < self.node_input_degree
        scale = np.sqrt(1.0 / (self.input_dimension * self.node_input_degree))
        self.input_weights = np.where(connect_mask, self.rng.normal(0, scale, (self.number_nodes, self.input_dimension)), 0)

    def one_step(self, inputs, standardize=False):
        """
        Advances the reservoir's state by one time step using the current state, input data, and reservoir matrices.
        
        Parameters:
            inputs (np.ndarray): Input data vector for the current time step.
        
        Returns:
            np.ndarray: The updated state of the reservoir.
        """
        input_term = self.input_weights @ inputs
        state_term = self.adjacency_weight_matrix @ self.state
        self.state = np.tanh(state_term + input_term + self.biases)

        if standardize:
            self.reapply_standardize()

        return self.state

    def run_given_inputs(self, inputs, calc_standardize=True):
        """
        Processes multiple inputs through the reservoir, updating states for each input and caching the results.
        
        Parameters:
            inputs (np.ndarray): A 2D array where each row corresponds to an input vector to the reservoir.
        """
        self.states = np.zeros((len(inputs), self.number_nodes))
        for i, input_vector in enumerate(inputs):
            self.states[i, :] = self.one_step(input_vector)

        if self.standardize:
            if calc_standardize:
                self.calculate_standardize_and_apply()
            else:
                self.reapply_standardize()

    def calculate_standardize_and_apply(self):
        """
        Standardizes the state sequences of the reservoir by subtracting the mean and dividing by the standard deviation,
        adjusting for each node across all states.
        """
        if self.states.size == 0:
            raise ValueError("No states available to standardize.")
        self.col_means = np.mean(self.states, axis=0)
        self.col_stds = np.std(self.states, axis=0)

        # To avoid division by zero, replace zero standard deviations with 1
        self.col_stds[self.col_stds == 0] = 1

        self.states_stand = (self.states - self.col_means) / self.col_stds

    def reapply_standardize(self):
        """
        Reapplies previously computed standardization parameters to new state sequences.
        """
        if not hasattr(self, 'col_means') or not hasattr(self, 'col_stds'):
            raise ValueError("Standardization parameters are not available. Please calculate them first.")
        
        if self.states.size == 0:
            raise ValueError("No states available to reapply standardization.")
        
        # Reapply the mean and standard deviation adjustments
        self.states_stand = (self.states - self.col_means) / self.col_stds
        self.state_stand = (self.state - self.col_means) / self.col_stds