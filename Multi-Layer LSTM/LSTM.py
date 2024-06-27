import torch
import torch.nn as nn
import numpy as np
from LSTM_Cell import LSTMCell

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_layers (int): Number of LSTM layers
            bias (bool): Whether to use bias in LSTM cells
            output_size (int): Size of output
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        # Create a list to hold LSTM cells
        self.lstm_cell_list = nn.ModuleList()
        
        # First LSTM cell takes input_size as input
        self.lstm_cell_list.append(LSTMCell(self.input_size, self.hidden_size, self.bias))
        
        # Subsequent LSTM cells take hidden_size as input
        for l in range(1, self.num_layers):
            self.lstm_cell_list.append(LSTMCell(self.hidden_size, self.hidden_size, self.bias))

        # Final fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        """
        Forward pass of the LSTM.
        
        Args:
            input: Input tensor of shape (batch_size, sequence length, input_size)
            hx: Initial hidden state and cell state (optional)
        
        Returns:
            out: Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state if not provided
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()
            else:
                h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        else:
            h0 = hx

        outs = []
        hidden = list()
        
        # Initialize hidden and cell states for each layer
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        # Process each time step
        for t in range(input.size(1)):
            # Process each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer takes input from the sequence
                    hidden_l = self.lstm_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0], hidden[layer][1])
                    )
                else:
                    # Subsequent layers take input from the previous layer
                    hidden_l = self.lstm_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                    )
                hidden[layer] = hidden_l
            
            # Store the output of the last layer
            outs.append(hidden_l[0])

        # Take only the last time step
        out = outs[-1].squeeze()
        
        # Pass through final fully connected layer
        out = self.fc(out)

        return out