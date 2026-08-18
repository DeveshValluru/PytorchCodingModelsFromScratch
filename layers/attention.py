import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
  def __init__(self, dropout :float =0.1):

    super().__init__() 
    self.dropout = nn.Dropout(dropout)


  def forward(self,query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))
    scores = scores/math.sqrt(d_k)

    if mask is not None:
      scores = scores.masked_fill(mask==0, float('-inf'))

    attention_weights = F.softmax(scores, dim =-1)
    attention_weights = self.dropout(attention_weights)

    output = torch.matmul(attention_weights, value)

    return output, attention_weights



class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads: int, d_model: int, dropout: float =0.1):
    super().__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_k = d_model // n_heads

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    self.w_o = nn.Linear(d_model,d_model)

    self.attention = ScaledDotProductAttention(dropout)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask:torch.Tensor = None):
    batch_size = query.size(0)
    Q = self.w_q(query)
    K = self.w_k(key)
    V = self.w_v(value)

    Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
    K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
    V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)

    output, attention_weights = self.attention(Q,K,V,mask)

    output = output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

    output = self.w_o(output)

    return output,attention_weights
