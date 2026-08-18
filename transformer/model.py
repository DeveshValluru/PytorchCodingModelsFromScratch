
import torch
import torch.nn as nn

from transformer.encoder import Encoder
from transformer.decoder import Decoder

class Transformer(nn.Module):
  def __init__(self,src_vocab_size, tgt_vocab_size, n_layers, d_model,d_ff, n_heads, max_len, dropout:float = 0.1):
    super().__init__()
    self.encoder = Encoder(vocab_size=src_vocab_size, d_model=d_model, d_ff=d_ff, n_layers=n_layers, n_heads=n_heads, max_len=max_len, dropout=dropout)
    self.decoder = Decoder(vocab_size=tgt_vocab_size, n_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, max_len=max_len, dropout=dropout)
    self.output_linear = nn.Linear(d_model,tgt_vocab_size)


  def forward(self, src:torch.Tensor, tgt: torch.Tensor, pad_tokens=0)-> torch.Tensor:
    src_mask = self.generate_src_mask(src,pad_tokens)
    tgt_mask = self.generate_tgt_mask(tgt,pad_tokens)
    encoder_output = self.encoder(src,src_mask)

    decoder_output = self.decoder(tgt,encoder_output, src_mask,tgt_mask)

    output =self.output_linear(decoder_output)
    return output

  def generate_src_mask(self,src,pad_tokens=0):
    return (src !=pad_tokens).unsqueeze(1).unsqueeze(2)

  def generate_tgt_mask(self,tgt, pad_tokens=0):
    tgt_pad_mask = (tgt != pad_tokens).unsqueeze(1).unsqueeze(2)

    seq_len = tgt.size(1)
    causal_mask = torch.tril(torch.ones(seq_len,seq_len, device = tgt.device)).bool()
    tgt_mask = tgt_pad_mask & causal_mask
    return tgt_mask


  


