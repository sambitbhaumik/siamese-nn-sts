import torch
import copy
import torch.nn as nn
from positional_encoding import PositionalEncoder

def clone_layer(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def similarity_score(input1, input2):
    # Get similarity predictions:
    dif = input1.squeeze() - input2.squeeze()

    norm = torch.norm(dif, p=1, dim=dif.dim() - 1)
    y_hat = torch.exp(-norm)
    y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
    return y_hat
