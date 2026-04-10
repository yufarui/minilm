import torch

from src.model.attention import Attention
from src.model.rotary_embedding import RotaryEmbedding


def test_attention_repeat_kv_shape():
    x = torch.randn(2, 2, 5, 8)
    y = Attention.repeat_kv(x, num_repeat=3)
    assert y.shape == (2, 6, 5, 8)


def test_attention_forward_cpu_eager(tiny_config):
    attn = Attention(tiny_config, layer_idx=0)
    rope = RotaryEmbedding(tiny_config)
    attn.eval()
    bsz, seqlen = 2, 6
    x = torch.randn(bsz, seqlen, tiny_config.hidden_size)

    position_ids = torch.arange(seqlen).unsqueeze(0).repeat(bsz, 1)
    cos, sin = rope(x, position_ids)

    mask = torch.ones(bsz, 1, seqlen, seqlen, dtype=torch.long)
    mask[:, :, :, -1] = 0

    y, w = attn(x, position_embeddings=(cos, sin), attention_mask=mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert w is not None
    assert torch.isfinite(w).all()
