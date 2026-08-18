import torch
import torch.nn as nn

from layers.feed_forward import FeedForward
from layers.positional_encoding import PositionalEncoding

from layers.attention import MultiHeadAttention



class DecoderBlock(nn.Module):
  def __init__(self,n_heads, d_model, d_ff, dropout:float = 0.1):
    super().__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = nn.Dropout(dropout)

    self.self_attn = MultiHeadAttention(n_heads,d_model,dropout)
    self.cross_attn = MultiHeadAttention(n_heads,d_model,dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.feed_forward = FeedForward(d_model,d_ff,dropout)

  def forward(self, x: torch.Tensor,encoder_output, src_mask: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
    self_attn_output,_ = self.self_attn(x,x,x,tgt_mask)
    self_attn_output = self.dropout(self_attn_output)
    x = self.norm1(self_attn_output + x)
    cross_attn_output,_ = self.cross_attn(x,encoder_output,encoder_output,src_mask)
    cross_attn_output = self.dropout(cross_attn_output)
    x = self.norm2(cross_attn_output + x)
    ff_output = self.feed_forward(x)
    ff_output = self.dropout(ff_output)
    output = self.norm3(ff_output + x)

    return output




class Decoder(nn.Module):
  def __init__(self,vocab_size, n_layers, d_model,d_ff, n_heads, max_len, dropout:float = 0.1):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size,d_model)
    self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
    self.layers = nn.ModuleList([
        DecoderBlock(n_heads,d_model,d_ff,dropout)
        for _ in range(n_layers)
    ])

  def forward(self,x:torch.Tensor, encoder_output:torch.Tensor, src_mask:torch.Tensor = None, tgt_mask: torch.Tensor = None)-> torch.Tensor:
    embeddings = self.embedding(x)
    pos_embeddings = self.pos_enc(embeddings)
    x = pos_embeddings

    for layer in self.layers:
      x = layer(x,encoder_output,src_mask, tgt_mask)
    return x

