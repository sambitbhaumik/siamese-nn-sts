import torch
import torch.nn as nn

from feed_forward import FeedForward
from layer_norm import LayerNorm
from multi_head_attention import MultiHeadAttention
from utils import clone_layer

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        # Self Attention followed by Feedforward
        self.d_model = d_model
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(n_heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.dropout_2 = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Self Attention followed by Feed Forward (Source for below snippet : http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks)
        x = x + self.dropout_1(self.multi_head_attention(x, x, x, mask))
        x = self.norm_1(x)
        x = x + self.dropout_2(self.feed_forward(x))
        x = self.norm_2(x)
        return x

class SublayerConnection(nn.Module):
    
    # Residual connection after which we have to apply Layer Norm 
    def __init__(self, size, dropout):
        
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # Apply residual connection to sublayer
        return x + self.dropout(sublayer(self.norm(x)))
