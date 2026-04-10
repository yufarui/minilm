import torch

from src.config.model_config import MiniLMConfig
from src.model.mlp import MLP
from src.model.rms_norm import RMSNorm
from src.model.rotary_embedding import RotaryEmbedding


def test_rms_norm_shape_dtype_and_finite(tiny_config):
    mod = RMSNorm(tiny_config)
    x = torch.randn(2, 5, tiny_config.hidden_size, dtype=torch.float32, requires_grad=True)
    y = mod(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_rms_norm_backward(tiny_config):
    mod = RMSNorm(tiny_config)
    x = torch.randn(2, 4, tiny_config.hidden_size, dtype=torch.float32, requires_grad=True)
    y = mod(x).sum()
    y.backward()
    assert x.grad is not None
    assert mod.weight.grad is not None


def test_mlp_forward_shape_and_finite(tiny_config):
    mlp = MLP(tiny_config)
    x = torch.randn(2, 6, tiny_config.hidden_size)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mlp_backward_has_grads(tiny_config):
    mlp = MLP(tiny_config)
    x = torch.randn(2, 3, tiny_config.hidden_size, requires_grad=True)
    loss = mlp(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None


def test_rotary_embedding_shapes_dtype_finite(tiny_config):
    rope = RotaryEmbedding(tiny_config)
    x = torch.randn(2, 7, tiny_config.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(7).unsqueeze(0).repeat(2, 1)
    cos, sin = rope(x, position_ids)
    head_dim = tiny_config.hidden_size // tiny_config.num_attention_heads
    assert cos.shape == (2, 7, head_dim)
    assert sin.shape == (2, 7, head_dim)
    assert cos.dtype == x.dtype
    assert sin.dtype == x.dtype
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()


def test_rotary_embedding_yarn_path_runs(tokenizer_vocab_size):
    cfg = MiniLMConfig(
        vocab_size=tokenizer_vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        inference_rope_scaling=True,
        max_position_embeddings=64,
    )
    rope = RotaryEmbedding(cfg)
    x = torch.randn(1, 5, cfg.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(5).unsqueeze(0)
    cos, sin = rope(x, position_ids)
    assert cos.shape[1] == 5 and sin.shape[1] == 5
