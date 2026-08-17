# PyTorch - Models From Scratch

Implementing neural network architectures from scratch in PyTorch for deep understanding and ML interview preparation.

## Architectures

### Transformer (Encoder-Decoder) — August 16, 2026

Full implementation of the original "Attention Is All You Need" architecture.

**Components:**
- `transformer/attention.py` — Scaled Dot-Product Attention, Multi-Head Attention
- `transformer/layers.py` — Feed-Forward Network, Positional Encoding (sinusoidal)
- `transformer/encoder.py` — Encoder Block, Encoder (N stacked blocks)
- `transformer/decoder.py` — Decoder Block (masked self-attention + cross-attention), Decoder
- `transformer/model.py` — Full Transformer with source/target mask generation

**Run:**
```bash
python test_transformer.py
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch
```
