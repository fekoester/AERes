import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler


def create_node_trajectory_windows(states, window_size):
    """
    Converts reservoir states into node-trajectory tokens.

    Args:
    - states (numpy.ndarray): Reservoir states with shape (time_steps, number_nodes).
    - window_size (int): Number of recent time steps included in each node trajectory.

    Returns:
    - numpy.ndarray: X with shape (samples, number_nodes, window_size).
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError("states must have shape (time_steps, number_nodes).")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if len(states) < window_size:
        raise ValueError("window_size cannot be larger than the number of state time steps.")

    trajectory_windows = []
    for end_idx in range(window_size - 1, len(states)):
        window = states[end_idx - window_size + 1:end_idx + 1]
        trajectory_windows.append(window.T)

    return np.asarray(trajectory_windows, dtype=np.float32)


def create_node_trajectory_dataset(states, targets, window_size):
    """
    Converts reservoir states and aligned targets into node-trajectory training samples.
    """
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if len(states) != len(targets):
        raise ValueError("states and targets must have the same number of time steps.")

    X = create_node_trajectory_windows(states, window_size)
    Y = targets[window_size - 1:]
    return X, Y

class LinearAttention(nn.Module):
    """
    A PyTorch module for applying a linear attention mechanism across an input feature set.
    Each target dimension has its own attention computation which is done through separate linear layers.

    Attributes:
        input_dim (int): Dimensionality of the input features.
        target_dim (int): Number of output targets.
        device (torch.device): The device tensors will be allocated on.

    Methods:
        forward(x): Defines the forward pass of the LinearAttention model.
    """
    def __init__(self, input_dim, target_dim, device=None):
        """
        Initializes the LinearAttention model.

        Args:
        - input_dim (int): Dimensionality of the input features.
        - target_dim (int): Number of output targets, each will have its own attention mechanism.
        - device (str or torch.device): The device tensors will be allocated on. If None, uses CUDA if available.
        """
        super(LinearAttention, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.target_dim = target_dim
        
        self.transform = nn.Linear(input_dim, input_dim * target_dim)
        self.to(self.device)


    def forward(self, x):
        """
        Defines the forward pass of the LinearAttention model.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying linear attention.
        """
        # Transform x in a single batch operation
        transformed = self.transform(x)  # Shape: (batch_size, input_dim * target_dim)

        # Reshape and sum across the appropriate dimensions
        transformed = transformed.view(x.size(0), self.target_dim, self.input_dim)
        outputs = torch.sum(transformed * x.unsqueeze(1), dim=2) / self.input_dim

        return outputs

class DeepLinearAttention(nn.Module):
    """
    A PyTorch module for applying a nonlinear attention mechanism.
    """
    def __init__(self, input_dim, target_dim, device=None):
        """
        Initializes the DeepLinearAttention model.

        Args:
        - input_dim (int): Dimensionality of the input features.
        - target_dim (int): Number of output targets, each will have its own attention mechanism.
        - device (str or torch.device): The device tensors will be allocated on. If None, uses CUDA if available.
        """
        super(DeepLinearAttention, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.target_dim = target_dim

        # Define a deep nonlinear layer with tanh activation
        self.deep_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        self.transform = nn.Linear(input_dim, input_dim * target_dim)
        self.to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the DeepLinearAttention model.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying linear attention.
        """
        # Pass input through the deep nonlinear layer
        x = self.deep_layer(x)

        # Transform x in a single batch operation
        transformed = self.transform(x)  # Shape: (batch_size, input_dim * target_dim)

        # Reshape and sum across the appropriate dimensions
        transformed = transformed.view(x.size(0), self.target_dim, self.input_dim)
        outputs = torch.sum(transformed * x.unsqueeze(1), dim=2) / self.input_dim

        return outputs


class NodeTrajectoryAttention(nn.Module):
    """
    Self-attention over reservoir nodes, where each node token is its recent state trajectory.
    """
    def __init__(self, number_nodes, window_size, target_dim, d_model=64, nhead=4,
                 num_layers=1, dim_feedforward=128, dropout=0.0, pooling="mean",
                 device=None):
        """
        Args:
        - number_nodes (int): Number of reservoir nodes, used as the attention token count.
        - window_size (int): Length of each node trajectory token.
        - target_dim (int): Number of output targets.
        - d_model (int): Token embedding dimension.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of Transformer encoder blocks.
        - dim_feedforward (int): Hidden size inside each encoder block.
        - dropout (float): Dropout used in encoder blocks.
        - pooling (str): One of "mean", "max", "flatten", "attention", or "cls".
        - device (str or torch.device): Device for model parameters.
        """
        super(NodeTrajectoryAttention, self).__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        valid_pooling = {"mean", "max", "flatten", "attention", "cls"}
        if pooling not in valid_pooling:
            raise ValueError(f"pooling must be one of {sorted(valid_pooling)}.")

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.number_nodes = number_nodes
        self.window_size = window_size
        self.target_dim = target_dim
        self.d_model = d_model
        self.pooling = pooling

        self.token_projection = nn.Linear(window_size, d_model)
        self.node_embedding = nn.Parameter(torch.zeros(1, number_nodes, d_model))

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pool_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if pooling == "flatten":
            readout_input_dim = number_nodes * d_model
        else:
            readout_input_dim = d_model
        self.readout = nn.Linear(readout_input_dim, target_dim)

        nn.init.normal_(self.node_embedding, std=0.02)
        if pooling == "cls":
            nn.init.normal_(self.cls_token, std=0.02)
        if pooling == "attention":
            nn.init.normal_(self.pool_query, std=0.02)

        self.to(self.device)

    def forward(self, x):
        """
        Args:
        - x (torch.Tensor): Shape (batch_size, number_nodes, window_size).

        Returns:
        - torch.Tensor: Shape (batch_size, target_dim).
        """
        if x.dim() != 3:
            raise ValueError("NodeTrajectoryAttention expects input with shape (batch, nodes, window_size).")
        if x.size(1) != self.number_nodes or x.size(2) != self.window_size:
            raise ValueError(
                f"Expected input shape (*, {self.number_nodes}, {self.window_size}), "
                f"got {tuple(x.shape)}."
            )

        tokens = self.token_projection(x) + self.node_embedding
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        encoded = self.encoder(tokens)

        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        elif self.pooling == "max":
            pooled = encoded.max(dim=1).values
        elif self.pooling == "flatten":
            pooled = encoded.reshape(encoded.size(0), -1)
        elif self.pooling == "attention":
            query = self.pool_query.expand(x.size(0), -1, -1)
            pooled = self.pool_attention(query, encoded, encoded)[0].squeeze(1)
        elif self.pooling == "cls":
            pooled = encoded[:, 0]

        return self.readout(pooled)

class AttentionTrainer:
    """
    Trainer class for the LinearAttention model, encapsulating training and evaluation logic,
    including automatic DataLoader creation from provided datasets.
    """
    def __init__(self, X, Y, batch_size=64, shuffle=True, learning_rate=0.001, use_gpu=True, layer_type="linear"):
        """
        Initializes the LinearAttentionTrainer with data and constructs the DataLoader.

        Args:
        - X_train (numpy.ndarray): Input feature array for training.
        - Y_train (numpy.ndarray): Output target array for training.
        - batch_size (int): Batch size for the DataLoader.
        - shuffle (bool): Whether to shuffle the data before training.
        - learning_rate (float): Learning rate for the optimizer.
        - device (str or torch.device): The device computations will run on. If None, uses CUDA if available.
        """
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if layer_type == "linear":
            self.model = LinearAttention(X.shape[1], Y.shape[1], device=self.device)
        if layer_type == "nonlinear":
            self.model = DeepLinearAttention(X.shape[1], Y.shape[1], device=self.device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = batch_size
        self.train_loader = self.create_dataloader(X, Y, batch_size, shuffle)
        self.mse_loss = nn.MSELoss()
        #self.writer = SummaryWriter(log_dir='./logs')  # Initialize the SummaryWriter

    def MSELoss(self, prediction, target):
        """
        Computes the mean squared error between the predicted and target values.

        Args:
            prediction (torch.Tensor or numpy.ndarray): Predicted values from the model.
            target (torch.Tensor or numpy.ndarray): Actual target values to compare against.

        Returns:
            torch.Tensor: The computed mean squared error loss.
        """
        # Convert numpy arrays to torch tensors
        if not isinstance(prediction, torch.Tensor):
            prediction = torch.tensor(prediction, dtype=torch.float32)

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        return self.mse_loss(prediction, target)
    
    def create_dataloader(self, X, Y, batch_size, shuffle=True):
        """
        Helper method to create a DataLoader from numpy arrays.

        Args:
        - X (numpy.ndarray): Input feature array.
        - Y (numpy.ndarray): Output target array.
        - batch_size (int): Batch size for the DataLoader.
        - shuffle (bool): Whether to shuffle the data.

        Returns:
        - DataLoader: Configured DataLoader ready for model training or evaluation.
        """
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(Y, dtype=torch.float32))
        num_workers = 8 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=0, pin_memory=pin_memory)

    def train(self, epochs=100):
        """
        Trains the model for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the model.
        """
        if self.device.type == "cuda":
            scaler = GradScaler()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_idx, (X_batch, Y_batch) in enumerate(self.train_loader):
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                self.optimizer.zero_grad()

                if self.device.type == "cuda":
                    with autocast():
                        predictions = self.model(X_batch)
                        loss = self.MSELoss(predictions, Y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    predictions = self.model(X_batch)
                    loss = self.MSELoss(predictions, Y_batch)
                    loss.backward()
                    self.optimizer.step()
                

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.train_loader):.4f}")



    def evaluate(self, X, Y):
        """
        Evaluates the model using a provided DataLoader for test data.

        Args:
        - test_loader (DataLoader): DataLoader for test data.
        """
        self.test_loader = self.create_dataloader(X, Y, self.batch_size)
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in self.test_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.MSELoss(outputs, Y_batch)
                total_loss += loss.item()

            print(f'Test Loss: {total_loss/len(self.test_loader):.4f}')

    def print_memory_summary(self):
        """
        Prints a summary of memory allocation and usage by the current model on the GPU.
        """
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    def predict(self, input):
        """
        Takes an input and makes a prediction on it.
        """
        self.model.to(self.device)
        input_tensor = torch.tensor(input, dtype=torch.float32).to(self.device)
        return self.model(input_tensor).detach().cpu().numpy()

    def save_model(self, filepath):
        """
        Saves the model parameters to the specified file.
        
        Args:
        - filepath (str): The path to the file where the model state will be saved.
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Loads model parameters from the specified file.
        
        Args:
        - filepath (str): The path to the file from which to load the model state.
        """
        model_state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(model_state)
        print(f"Model loaded from {filepath}")


class NodeTrajectoryAttentionTrainer:
    """
    Trainer for self-attention over reservoir node trajectories.
    """
    def __init__(self, X, Y, window_size=None, batch_size=64, shuffle=True,
                 learning_rate=0.001, use_gpu=True, d_model=64, nhead=4,
                 num_layers=1, dim_feedforward=128, dropout=0.0, pooling="mean"):
        """
        Args:
        - X (numpy.ndarray): Either raw reservoir states (time, nodes) or node trajectories
          (samples, nodes, window_size).
        - Y (numpy.ndarray): Targets aligned with X.
        - window_size (int): Required when X contains raw reservoir states.
        - pooling (str): One of "mean", "max", "flatten", "attention", or "cls".
        """
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size

        if X.ndim == 2:
            if window_size is None:
                raise ValueError("window_size is required when X contains raw reservoir states.")
            X, Y = create_node_trajectory_dataset(X, Y, window_size)
        elif X.ndim == 3:
            if window_size is None:
                window_size = X.shape[2]
        else:
            raise ValueError("X must have shape (time, nodes) or (samples, nodes, window_size).")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.window_size = window_size
        self.train_loader = self.create_dataloader(X, Y, batch_size, shuffle)
        self.model = NodeTrajectoryAttention(
            number_nodes=X.shape[1],
            window_size=X.shape[2],
            target_dim=Y.shape[1],
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling,
            device=self.device,
        )
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mse_loss = nn.MSELoss()

    def MSELoss(self, prediction, target):
        if not isinstance(prediction, torch.Tensor):
            prediction = torch.tensor(prediction, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        return self.mse_loss(prediction, target)

    def create_dataloader(self, X, Y, batch_size, shuffle=True):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        pin_memory = self.device.type == "cuda"
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin_memory)

    def train(self, epochs=100):
        if self.device.type == "cuda":
            scaler = GradScaler()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, Y_batch in self.train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                self.optimizer.zero_grad()

                if self.device.type == "cuda":
                    with autocast():
                        predictions = self.model(X_batch)
                        loss = self.MSELoss(predictions, Y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    predictions = self.model(X_batch)
                    loss = self.MSELoss(predictions, Y_batch)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.train_loader):.4f}")

    def evaluate(self, X, Y, window_size=None):
        if X.ndim == 2:
            if window_size is None:
                window_size = self.window_size
            X, Y = create_node_trajectory_dataset(X, Y, window_size)

        test_loader = self.create_dataloader(X, Y, self.batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.MSELoss(outputs, Y_batch)
                total_loss += loss.item()

        print(f'Test Loss: {total_loss/len(test_loader):.4f}')

    def predict(self, input, window_size=None):
        if input.ndim == 2:
            if input.shape[0] == self.model.number_nodes and input.shape[1] == self.window_size:
                input = input.reshape(1, self.model.number_nodes, self.window_size)
            else:
                if window_size is None:
                    window_size = self.window_size
                input = create_node_trajectory_windows(input, window_size)

        self.model.eval()
        input_tensor = torch.tensor(input, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(input_tensor).detach().cpu().numpy()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        model_state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(model_state)
        print(f"Model loaded from {filepath}")
