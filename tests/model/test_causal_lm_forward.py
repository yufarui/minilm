import torch

from src.model.model import MiniLMModel
from src.model.model import MiniLmForCausalLM


def test_prepare_attention_mask_defaults_to_causal():
    mask = MiniLMModel._prepare_autoregressive_attention_mask(
        attention_mask=None,
        batch_size=2,
        query_length=4,
        key_length=4,
        past_seen_tokens=0,
        device=torch.device("cpu"),
    )
    expected = torch.tril(torch.ones(2, 1, 4, 4, dtype=torch.bool))
    assert torch.equal(mask, expected)


@torch.no_grad()
def test_no_attention_mask_matches_all_valid_causal_mask(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    input_ids = torch.randint(1, tiny_config.vocab_size, (2, 6))
    explicit_attention_mask = torch.ones_like(input_ids)

    no_mask_logits = model(input_ids=input_ids, use_cache=False).logits
    explicit_mask_logits = model(
        input_ids=input_ids,
        attention_mask=explicit_attention_mask,
        use_cache=False,
    ).logits

    assert torch.allclose(no_mask_logits, explicit_mask_logits, atol=1e-6, rtol=1e-6)


def test_causal_lm_eval_forward_and_logits_slice(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    out = model(input_ids=input_ids, use_cache=True, logits_to_keep=2)
    assert out.logits.shape == (2, 2, tiny_config.vocab_size)
    assert out.past_key_values is not None
    assert torch.isfinite(out.logits).all()


def test_causal_lm_train_returns_loss(tiny_config):
    model = MiniLmForCausalLM(tiny_config).train()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 5))
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
