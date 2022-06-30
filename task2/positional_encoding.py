import math

import torch
import torch.nn as nn

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant positional encoding matrix
        # Make a matrix of [Sequence Length , Hidden Dimension] representing the positional encoding for max_len inputs
        pe_matrix = torch.zeros(max_seq_len, d_model)
        
        # Calculate the feature pattern proposed in the Vaswani Paper
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe_matrix[pos, i] = math.sin(pos/10000**(2*i/d_model))
                pe_matrix[pos, i+1] = math.cos(pos/10000**(2*i/d_model))
        pe_matrix = pe_matrix.unsqueeze(0)     # Add one dimension for batch size

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state. (Source : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe_matrix)  # Register as persistent buffer
        
    def forward(self, x):
        # x is a sentence after embedding with dim (batch, number of words, vector dimension)
        seq_len = x.size()[1]
        x = x + self.pe[:, :seq_len]
        return x