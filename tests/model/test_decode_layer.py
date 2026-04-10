import torch

from src.model.decode_layer import DecoderLayer


def test_decode_layer_forward_and_backward(tiny_config):
    layer = DecoderLayer(tiny_config, layer_idx=0)
    layer.train()
    bsz, seqlen = 2, 5
    x = torch.randn(bsz, seqlen, tiny_config.hidden_size, requires_grad=True)
    head_dim = tiny_config.hidden_size // tiny_config.num_attention_heads
    cos = torch.ones(bsz, seqlen, head_dim)
    sin = torch.zeros(bsz, seqlen, head_dim)
    mask = torch.ones(bsz, 1, seqlen, seqlen, dtype=torch.long)

    y, _ = layer(x, attention_mask=mask, position_embeddings=(cos, sin))
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None
