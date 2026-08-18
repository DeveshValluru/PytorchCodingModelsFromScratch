import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len,dropout : float = 0.1):
    super().__init__()
    self.d_model = d_model
    self.max_len = max_len
    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0,max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0,d_model,2).float() * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(pos * div_term)

    pe[:, 1::2] = torch.cos(pos * div_term)
    pe = pe.unsqueeze(0)

    self.register_buffer('pe', pe)



  def forward(self, x):
    seq_len = x.size(1)
    pe = self.pe[:, :seq_len, :]

    result = x + pe

    return self.dropout(result)
