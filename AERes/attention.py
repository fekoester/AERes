import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

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
    A PyTorch module for applying a linear attention mechanism with an additional deep nonlinear layer.
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

class LinearAttentionTrainer:
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
        self.scaler = GradScaler()
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


    def create_dataloader(self, X, Y, batch_size, shuffle=False):
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
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, persistent_workers=True)
    


    def train(self, epochs=100):
        """
        Trains the model for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the model.
        """
        self.model.train()
        self.model.to(self.device)
      
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (X_batch, Y_batch) in enumerate(self.train_loader):
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                self.optimizer.zero_grad()

                if self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(X_batch)
                        loss = self.MSELoss(outputs, Y_batch)
                else:
                    outputs = self.model(X_batch)
                    loss = self.MSELoss(outputs, Y_batch)

                if self.device.type == 'cuda':
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            if epoch % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')


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