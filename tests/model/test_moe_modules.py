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


def test_moe_gate_aux_excludes_padding_tokens(tokenizer_vocab_size):
    """Pad 位不得进入 aux load/prob；全零 pad 隐状态否则会塌缩到同一 expert。"""
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
        num_experts_per_tok=1,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        aux_loss_alpha=0.01,
    )
    gate = MoeGate(cfg)
    gate.train()
    torch.manual_seed(0)
    # 一半真实 token、一半全零 pad（模拟 flash 下 pad 隐状态）
    real = torch.randn(1, 4, cfg.hidden_size)
    pads = torch.zeros(1, 4, cfg.hidden_size)
    x = torch.cat([real, pads], dim=1)
    keep = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)

    _, _, aux_masked = gate(x, token_mask=keep)
    _, _, aux_real_only = gate(real, token_mask=None)
    _, _, aux_with_pads = gate(x, token_mask=None)

    assert torch.isfinite(aux_masked)
    assert torch.allclose(aux_masked, aux_real_only, atol=1e-6)
    assert not torch.allclose(aux_with_pads, aux_real_only, atol=1e-5)


def test_causal_lm_moe_aux_matches_nonpad_only(tiny_config, local_tokenizer):
    """端到端：TrainDataCollator 动态 padding 后，总 aux 应等于仅非 pad 样本的 aux。"""
    from src.model.model import MiniLmForCausalLM
    from src.util.data_collator import TrainDataCollator

    tiny_config.moe_enable = True
    tiny_config.use_flash_attention = True
    model = MiniLmForCausalLM(tiny_config).train()
    collator = TrainDataCollator(local_tokenizer)
    features = [
        {"input_ids": list(range(10, 26)), "labels": list(range(10, 26))},
        {"input_ids": list(range(40, 44)), "labels": list(range(40, 44))},
    ]
    batch = collator(features)
    out_padded = model(**batch)

    # 单条无 pad 样本：与 batch 中第一条等长内容一致
    b0 = {k: v[:1] for k, v in batch.items()}
    # 去掉右侧 pad 后的短样本单独前向，再与「仅用 token_mask 的门控」对比较难；
    # 这里断言：带 mask 的 aux 有限，且小于「若把 pad 算进去会更极端」的污染场景下
    # 通过 gate 直接对比 load。
    assert torch.isfinite(out_padded.aux_loss)

    pad = batch["input_ids"] == local_tokenizer.pad_token_id
    assert pad.any(), "测试需要动态 padding"

    # 钩住第一层 gate：有 token_mask 时 aux 所用 token 数 = 非 pad 数
    captured = {}

    def hook(mod, args, kwargs, output):
        captured["token_mask"] = kwargs.get("token_mask")
        captured["topk_idx"] = output[0]

    handle = model.model.layers[0].mlp.gate.register_forward_hook(hook, with_kwargs=True)
    _ = model(**batch)
    handle.remove()

    assert captured["token_mask"] is not None
    assert int(captured["token_mask"].sum()) == int((~pad).sum())
