"""NeRF model definition."""
import torch.nn as nn
import torch.nn.functional as F


class NeRFModel(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_hidden: int = 256,
        num_input_features: int = 5,
        num_output_features: int = 4,
        activation: str = "relu",
    ):
        """
        Initializes the NeRF model with specified parameters.

        Args:
            num_layers (int): Number of layers in the model (must be >= 2).
            num_hidden (int): Number of hidden units in each layer.
            num_input_features (int): Number of input features.
            num_output_features (int): Number of output features.
            activation (str): Activation function to use ("relu" or "leaky_relu").
        """
        super(NeRFModel, self).__init__()

        # Validate the number of layers
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.activation = activation.lower()

        # Define the layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_input_features, num_hidden))  # Input layer
        self.layers.extend([nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)])  # Hidden layers
        self.layers.append(nn.Linear(num_hidden, num_output_features))  # Output layer

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Applies Xavier initialization to the weights of the linear layers."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass for the NeRF model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_output_features).
        """
        for i, layer in enumerate(self.layers[:-1]):  # Apply activation to all layers except the last
            x = layer(x)
            if self.activation == "relu":
                x = F.relu(x)
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(x, negative_slope=0.01)
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")

        # No activation for the last layer
        x = self.layers[-1](x)
        return x