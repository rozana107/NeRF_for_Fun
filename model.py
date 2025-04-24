"""NeRF model definition."""
import torch
import torch.nn as nn


class NeRFModel(nn.Module):
 
    def __init__(self,
        num_layers: int = 3,
        num_hidden: int = 256,
        num_input_features: int = 5,
        num_output_features: int = 4,
        use_leaky_relu: bool = False):
        """
        Initializes the NeRF model with specified parameters.
        Args:
            num_layers (int): Number of layers in the model.
            num_hidden (int): Number of hidden units in each layer.
            num_input_features (int): Number of input features.
            num_output_features (int): Number of output features.
        """
        super(NeRFModel, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_leaky_relu = use_leaky_relu
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(self.num_input_features, self.num_hidden)])
        self.layers.extend([nn.Linear(self.num_hidden, self.num_hidden) for _ in range(num_layers - 2)])
        self.layers.extend([nn.Linear(self.num_hidden, self.num_output_features)])
    
    def forward(self, x):
        """Forward pass for the NeRF model.
        
        Args:
            x (torch.Tensor): Input Tensor of shape (batch_size, num_input_features).
        
        Returns:
            torch.Tensor: Output Tensor of shape (batch_size, num_output_features).
        """
        for i in range(self.num_layers - 1):
            if self.use_leaky_relu:
                x = torch.nn.leaky_relu(self.layers[i](x))
            else:       
                x = torch.relu(self.layers[i](x))          
        x = self.layers[-1](x)

        return x

