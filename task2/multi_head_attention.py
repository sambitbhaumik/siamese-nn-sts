import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clone_layer

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    # Shape of q and k are the same, both are (batch_size, seq_len, d_k)

    attention_scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(q.shape[-1])  # size (batch_size, seq_len, seq_len)
    
    # Apply mask to scores
    # <pad>
    if mask is not None:
        bs = q.shape[0]
        tgt_len = q.shape[1]
        seq_len = k.shape[1]
        
        exp_mask = mask[:, None, :].expand(bs, tgt_len, seq_len)
        attention_scores = attention_scores.masked_fill(exp_mask == 0, value=-float("Inf"))
        
    # Softmax applied to the last dimension
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    if dropout is not None:
        attention_weights = dropout(attention_weights)
        
    output = torch.matmul(attention_weights, v)
    return output
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_v = d_model//n_heads # Embedding dimension must be divisible by number of heads 
        
        # self attention linear layers
        # Linear layers for q, k, v vectors generation in different heads
        self.q_linear_layers = []
        self.k_linear_layers = []
        self.v_linear_layers = []
        for i in range(n_heads):
            self.q_linear_layers.append(torch.nn.Linear(d_model, self.d_k))
            self.k_linear_layers.append(torch.nn.Linear(d_model, self.d_k))
            self.v_linear_layers.append(torch.nn.Linear(d_model, self.d_v))
        
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(n_heads*self.d_v, d_model)
        
    def forward(self, q, k, v, mask=None):
        # We implement the forward function in Multi Headed Attention with the variables query, key and value 

        multi_head_attention_outputs = []
        for q_linear, k_linear, v_linear in zip(self.q_linear_layers,
                                                self.k_linear_layers,
                                                self.v_linear_layers):
            new_q = q_linear(q)  # size: (batch_size, seq_len, d_k)
            new_k = k_linear(k)  # size: (batch_size, seq_len, d_k)
            new_v = v_linear(v)  # size: (batch_size, seq_len, d_v)
            
            # Scaled Dot-Product attention
            # First we perform linear projections in batches 
            head_v = scaled_dot_product_attention(new_q, new_k, new_v, mask, self.dropout)  # (batch_size, seq_len, d_v)
            multi_head_attention_outputs.append(head_v)
            
        # Concatenate the heads 
        concat = torch.cat(multi_head_attention_outputs, -1)  # (batch_size, seq_len, n_heads*d_v)
        
        # Linear layer to get the shape 
        # transpose to get dimensions bs * h * sl * d_model
        output = self.out(concat)  # (batch_size, seq_len, d_model)
        
        return output