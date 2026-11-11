import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_2d_float32(array):
    array = np.asarray(array, dtype=np.float32)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def create_node_trajectory_windows(states, window_size):
    """
    Converts reservoir states into node-trajectory tokens.

    Args:
    - states (numpy.ndarray): Reservoir states with shape (time_steps, number_nodes).
    - window_size (int): Number of recent time steps included in each node trajectory.

    Returns:
    - numpy.ndarray: X with shape (samples, number_nodes, window_size), where
      samples = time_steps - window_size + 1.
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError("states must have shape (time_steps, number_nodes).")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if len(states) < window_size:
        raise ValueError("window_size cannot be larger than the number of state time steps.")

    # Strided view: (samples, nodes, window) without a Python loop, then one contiguous copy.
    windows = np.lib.stride_tricks.sliding_window_view(states, window_size, axis=0)
    return np.ascontiguousarray(windows)


def create_node_trajectory_dataset(states, targets, window_size):
    """
    Converts reservoir states and aligned targets into node-trajectory training samples.
    """
    targets = _as_2d_float32(targets)
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
        self.device = _resolve_device(device)
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
        self.device = _resolve_device(device)
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

        self.device = _resolve_device(device)
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

class _TorchTrainer:
    """Shared training / evaluation / persistence logic for the PyTorch readouts."""

    def _setup(self, use_gpu, batch_size, learning_rate):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mse_loss = nn.MSELoss()
        self.history = []

    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def MSELoss(self, prediction, target):
        """Mean squared error between prediction and target (numpy arrays or tensors)."""
        if not isinstance(prediction, torch.Tensor):
            prediction = torch.tensor(np.asarray(prediction), dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(np.asarray(target), dtype=torch.float32)
        return self.mse_loss(prediction, target)

    def create_dataloader(self, X, Y, batch_size, shuffle=True):
        """Creates a DataLoader from numpy arrays."""
        dataset = TensorDataset(torch.as_tensor(np.asarray(X, dtype=np.float32)),
                                torch.as_tensor(_as_2d_float32(Y)))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                          pin_memory=self.device.type == "cuda")

    def train(self, epochs=100, verbose=True):
        """
        Trains the model.

        Args:
            epochs (int): Number of epochs.
            verbose (bool): Print the mean loss after every epoch.

        Returns:
            list[float]: Mean training loss per epoch (also appended to ``self.history``).
        """
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, Y_batch in self.train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                Y_batch = Y_batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = self.mse_loss(self.model(X_batch), Y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss = self.mse_loss(self.model(X_batch), Y_batch)
                    loss.backward()
                    self.optimizer.step()
                epoch_loss += loss.item()

            mean_loss = epoch_loss / len(self.train_loader)
            losses.append(mean_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss:.4f}")
        self.history.extend(losses)
        return losses

    def _prepare_inputs(self, X, Y=None, window_size=None):
        return X, Y

    def evaluate(self, X, Y, verbose=True, **kwargs):
        """
        Computes the test MSE over all samples.

        Returns:
            float: Mean squared error.
        """
        X, Y = self._prepare_inputs(X, Y, **kwargs)
        loader = self.create_dataloader(X, Y, self.batch_size, shuffle=False)
        self.model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                total_loss += self.mse_loss(self.model(X_batch), Y_batch).item() * len(X_batch)
                count += len(X_batch)
        loss = total_loss / count
        if verbose:
            print(f'Test Loss: {loss:.4f}')
        return loss

    def predict(self, input, **kwargs):
        """Predicts outputs for the given input array. Returns a numpy array."""
        input, _ = self._prepare_inputs(input, None, **kwargs)
        self.model.eval()
        input_tensor = torch.as_tensor(np.asarray(input, dtype=np.float32), device=self.device)
        with torch.no_grad():
            return self.model(input_tensor).cpu().numpy()

    def print_memory_summary(self):
        """Prints a summary of GPU memory allocation."""
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    def save_model(self, filepath):
        """Saves the model parameters to ``filepath``."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads model parameters from ``filepath``."""
        model_state = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_state)
        print(f"Model loaded from {filepath}")


class AttentionTrainer(_TorchTrainer):
    """
    Trainer for the attention-enhanced reservoir readout (``LinearAttention`` or ``DeepLinearAttention``).
    """
    LAYER_TYPES = {"linear": LinearAttention, "nonlinear": DeepLinearAttention}

    def __init__(self, X, Y, batch_size=64, shuffle=True, learning_rate=0.001, use_gpu=True, layer_type="linear"):
        """
        Args:
        - X (numpy.ndarray): Reservoir states, shape (samples, nodes).
        - Y (numpy.ndarray): Targets, shape (samples, targets) or (samples,).
        - batch_size (int): Batch size.
        - shuffle (bool): Shuffle training data every epoch.
        - learning_rate (float): Adam learning rate.
        - use_gpu (bool): Use CUDA if available.
        - layer_type (str): "linear" or "nonlinear".
        """
        if layer_type not in self.LAYER_TYPES:
            raise ValueError(f"layer_type must be one of {sorted(self.LAYER_TYPES)}, got {layer_type!r}.")
        self._setup(use_gpu, batch_size, learning_rate)
        Y = _as_2d_float32(Y)
        self.model = self.LAYER_TYPES[layer_type](X.shape[1], Y.shape[1], device=self.device)
        self._init_optimizer()
        self.train_loader = self.create_dataloader(X, Y, batch_size, shuffle)

    def _prepare_inputs(self, X, Y=None, **kwargs):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X, Y


class NodeTrajectoryAttentionTrainer(_TorchTrainer):
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
        self._setup(use_gpu, batch_size, learning_rate)
        X = np.asarray(X)
        if X.ndim == 2:
            if window_size is None:
                raise ValueError("window_size is required when X contains raw reservoir states.")
            X, Y = create_node_trajectory_dataset(X, Y, window_size)
        elif X.ndim == 3:
            if window_size is None:
                window_size = X.shape[2]
        else:
            raise ValueError("X must have shape (time, nodes) or (samples, nodes, window_size).")
        Y = _as_2d_float32(Y)

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
        self._init_optimizer()

    def _prepare_inputs(self, X, Y=None, window_size=None):
        """
        Accepts raw states (time, nodes), a single trajectory (nodes, window_size) or
        trajectories (samples, nodes, window_size). Raw states are windowed and targets are aligned.
        """
        X = np.asarray(X)
        window_size = window_size or self.window_size
        if X.ndim == 2:
            if Y is None and X.shape == (self.model.number_nodes, self.window_size):
                return X.reshape(1, *X.shape), None
            if Y is not None:
                return create_node_trajectory_dataset(X, Y, window_size)
            return create_node_trajectory_windows(X, window_size), None
        return X, Y
