import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer.layers import FeedForward, PositionalEncoding
from transformer.attention import MultiHeadAttention


class EncoderBlock(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout: float = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)

    self.attention = MultiHeadAttention(n_heads, d_model, dropout)
    self.feed_forward = FeedForward(d_model, d_ff, dropout)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

  def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    attn_output, _ = self.attention(x, x, x, mask)
    attn_output = self.dropout(attn_output)
    x = self.norm1(attn_output + x)
    ff_output = self.feed_forward(x)
    ff_output = self.dropout(ff_output)
    x = self.norm2(ff_output + x)
    return x


class Encoder(nn.Module):
  def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, max_len, dropout: float = 0.1):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positionalencoding = PositionalEncoding(d_model, max_len)
    self.layers = nn.ModuleList([
        EncoderBlock(d_model, n_heads, d_ff, dropout)
        for _ in range(n_layers)
    ])

  def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    embeddings = self.embedding(x)
    pos_embeddings = self.positionalencoding(embeddings)

    x = pos_embeddings
    for layer in self.layers:
      x = layer(x, mask)
    return x
