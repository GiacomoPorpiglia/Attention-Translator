import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, dim, max_len=5000):
        """
        Args:
            dim: Dimension of the model (embedding size).
            max_len: Maximum sequence length to precompute positional encodings for.
        """
        super(PositionalEncoder, self).__init__()
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, dim)
        
        # Compute the positional encodings
        # [0, 1, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) ### [max_len, 1]
        # compute div_term, shape = [dim/2]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) 
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) 
        
        # Add an extra dimension for batch compatibility (1, max_len, dim)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (non-trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
  
        ### x: [B, T, embd_size]

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
