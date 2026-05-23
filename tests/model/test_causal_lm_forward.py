import torch

from src.model.model import MiniLMModel, MiniLmForCausalLM


@torch.no_grad()
def test_default_forward_is_causal_without_attention_mask(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    prefix = torch.randint(1, tiny_config.vocab_size, (1, 4))
    future = torch.randint(1, tiny_config.vocab_size, (1, 3))

    prefix_logits = model(input_ids=prefix, use_cache=False).logits
    extended_logits = model(
        input_ids=torch.cat([prefix, future], dim=1),
        use_cache=False,
    ).logits[:, : prefix.shape[1], :]

    assert torch.allclose(prefix_logits, extended_logits, atol=1e-5, rtol=1e-5)


def test_2d_attention_mask_blocks_padding_queries(tiny_config):
    mask = MiniLMModel._prepare_autoregressive_attention_mask(
        attention_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        input_ids=None,
        pad_token_id=tiny_config.pad_token_id,
        batch_size=1,
        query_length=4,
        key_length=4,
        past_seen_tokens=0,
        device=torch.device("cpu"),
    )[0, 0]

    assert mask[:, 2:].sum().item() == 0
    assert mask[2:, :].sum().item() == 0


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
