import torch
import torch.nn as nn
import numpy as np
from GRU_Cell import GRUCell

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        # Create a list of GRU cells
        self.rnn_cell_list = nn.ModuleList()
        # First layer has input_size as input dimension
        self.rnn_cell_list.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        # Subsequent layers have hidden_size as input dimension
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        
        # Final fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        # Input of shape (batch_size, sequence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        # Initialize hidden state if not provided
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()
            else:
                h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        else:
            h0 = hx

        outs = []
        hidden = list()
        
        # Initialize hidden states for each layer
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        # Process input sequence
        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    # First layer takes input from the input sequence
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    # Subsequent layers take input from the previous layer
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], hidden[layer])
                
                hidden[layer] = hidden_l
            
            # Store the output of the last layer
            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1].squeeze()
        
        # Pass through the final fully connected layer
        out = self.fc(out)

        return out