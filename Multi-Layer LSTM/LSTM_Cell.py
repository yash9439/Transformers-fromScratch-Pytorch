import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        Initialize the LSTM cell.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            bias (bool): Whether to use bias in linear layers
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Linear layer for input-to-hidden transformation
        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        # Linear layer for hidden-to-hidden transformation
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        """
        Forward pass of the LSTM cell.
        
        Args:
            input: Input tensor of shape (batch_size, input_size)
            hx: Tuple of (hidden state, cell state), each of shape (batch_size, hidden_size)
        
        Returns:
            hy: New hidden state tensor of shape (batch_size, hidden_size)
            cy: New cell state tensor of shape (batch_size, hidden_size)
        """
        # Initialize hidden and cell states if not provided
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
            hx = (hx, hx)
        
        # Unpack hidden and cell states
        hx, cx = hx
        
        # Compute gates
        gates = self.xh(input) + self.hh(hx)
        
        # Split gates into input, forget, cell, and output gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        
        # Apply activation functions to gates
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)
        
        # Update cell state
        cy = cx * f_t + i_t * g_t
        
        # Compute new hidden state
        hy = o_t * torch.tanh(cy)
        
        return (hy, cy)