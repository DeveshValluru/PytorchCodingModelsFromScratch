import torch
from transformer.model import Transformer


def test_transformer():
    # Hyperparameters
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    d_model = 64
    d_ff = 256
    n_heads = 4
    n_layers = 2
    max_len = 100

    model = Transformer(src_vocab_size, tgt_vocab_size, n_layers, d_model, d_ff, n_heads, max_len)

    # Fake input — batch of 2, source length 8, target length 6
    src = torch.randint(1, src_vocab_size, (2, 8))
    tgt = torch.randint(1, tgt_vocab_size, (2, 6))

    output = model(src, tgt)

    print(f"Source shape:  {src.shape}")        # (2, 8)
    print(f"Target shape:  {tgt.shape}")        # (2, 6)
    print(f"Output shape:  {output.shape}")     # (2, 6, 1200)

    assert output.shape == (2, 6, tgt_vocab_size), f"Expected (2, 6, {tgt_vocab_size}), got {output.shape}"
    print("\nTransformer test passed!")


if __name__ == "__main__":
    test_transformer()
