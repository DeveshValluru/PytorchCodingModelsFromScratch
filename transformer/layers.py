import torch
import torch.nn as nn
import torch.nn.functional as F
import math





class FeedForward(nn.Module):
  def __init__(self,d_model: int, d_ff: int, dropout: float = 0.1):
    super().__init__()
    self.linear1 = nn.Linear(d_model,d_ff)
    self.linear2 = nn.Linear(d_ff,d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x



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




