import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, activation=nn.Identity()):
        """
        LoRA-style adapter module:
        - input_dim: the dimension of input features
        - bottleneck_dim: the reduced dimension (rank)
        - activation: activation function between two linear layers
        """
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.activation = activation
        self.up_proj = nn.Linear(bottleneck_dim, input_dim, bias=False)

    def forward(self, x):
        # Add residual connection if desired
        return x + self.up_proj(self.activation(self.down_proj(x)))
