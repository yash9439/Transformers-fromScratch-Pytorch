import torch
import torch.nn as nn
import numpy as np

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        """
        Initialize the RNN cell.
        
        Args:
            input_size (int): Size of the input vector
            hidden_size (int): Size of the hidden state vector
            bias (bool): Whether to use bias in linear layers
            nonlinearity (str): Type of activation function ('tanh' or 'relu')
        """
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        # Validate the nonlinearity option
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        # Define linear transformations
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)  # Input to hidden
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)  # Hidden to hidden

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset and initialize cell parameters using uniform distribution.
        This helps in stabilizing the learning process.
        """
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        """
        Perform forward pass of the RNN cell.

        Args:
            input: Tensor of shape (batch_size, input_size)
            hx: Hidden state tensor of shape (batch_size, hidden_size), optional

        Returns:
            hy: New hidden state tensor of shape (batch_size, hidden_size)
        """
        # If no hidden state is provided, initialize with zeros
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)

        # Combine input and hidden state
        hy = self.x2h(input) + self.h2h(hx)

        # Apply nonlinearity
        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy