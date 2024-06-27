import torch
import torch.nn as nn
import numpy as np

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Define linear transformations for input-to-hidden and hidden-to-hidden
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly within [-1/sqrt(hidden_size), 1/sqrt(hidden_size)]
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        # Inputs:
        # input: of shape (batch_size, input_size)
        # hx: of shape (batch_size, hidden_size)
        # Output:
        # hy: of shape (batch_size, hidden_size)

        # Initialize hidden state with zeros if not provided
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)

        # Compute linear transformations
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        # Split the transformations into reset, update, and new gate components
        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        # Compute reset gate
        reset_gate = torch.sigmoid(x_reset + h_reset)
        
        # Compute update gate
        update_gate = torch.sigmoid(x_upd + h_upd)
        
        # Compute candidate hidden state
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        # Compute final hidden state
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy