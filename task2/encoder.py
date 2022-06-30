import torch
import torch.nn as nn

from encoder_layer import EncoderLayer
from feed_forward import FeedForward
from layer_norm import LayerNorm
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoder
from utils import clone_layer

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, N, n_heads):
        super().__init__()
        #self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        # Encoder is a stack of N Layers which we can vary  
        
        self.encoder_layers = clone_layer(EncoderLayer(d_model, n_heads), N)
        self.norm = LayerNorm(d_model)
        
    def forward(self, src, mask):
        #x = self.embed(src)
        x = self.pe(src)
        for encoder in self.encoder_layers:
            x = encoder(x, mask)
        return self.norm(x)

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, d_model, N=6, n_heads=8):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, n_heads)
        
    def forward(self, src, src_mask):

        # Taking the inputs into the encoder with the mask 
        encoder_output = self.encoder(src, src_mask)
        # output = self.linear(encoder_output)
        return encoder_output