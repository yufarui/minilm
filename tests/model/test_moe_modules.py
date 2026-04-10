import pytest
import torch

from src.config.model_config import MiniLMConfig
from src.model.moe import Moe
from src.model.moe_gate import MoeGate


def test_moe_gate_shapes_and_ranges(tiny_config):
    gate = MoeGate(tiny_config)
    gate.eval()
    x = torch.randn(2, 4, tiny_config.hidden_size)
    topk_idx, topk_weight, aux_loss = gate(x)
    n = 2 * 4
    assert topk_idx.shape == (n, tiny_config.num_experts_per_tok)
    assert topk_weight.shape == (n, tiny_config.num_experts_per_tok)
    assert topk_idx.min() >= 0
    assert topk_idx.max() < tiny_config.n_routed_experts
    assert aux_loss.ndim == 0


def test_moe_gate_norm_topk_prob_sum_to_one(tokenizer_vocab_size):
    cfg = MiniLMConfig(
        vocab_size=tokenizer_vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_enable=True,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        aux_loss_alpha=0.01,
    )
    gate = MoeGate(cfg)
    x = torch.randn(1, 5, cfg.hidden_size)
    _, w, _ = gate(x)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_moe_gate_invalid_scoring_raises(tokenizer_vocab_size):
    cfg = MiniLMConfig(
        vocab_size=tokenizer_vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_enable=True,
        scoring_func="invalid",
    )
    gate = MoeGate(cfg)
    x = torch.randn(1, 2, cfg.hidden_size)
    with pytest.raises(NotImplementedError):
        gate(x)


def test_moe_forward_shape_and_aux_loss(tiny_config):
    moe = Moe(tiny_config)
    moe.train()
    x = torch.randn(2, 4, tiny_config.hidden_size)
    y = moe(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(moe.aux_loss)
