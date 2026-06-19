import torch

from src.model.model import MiniLmForCausalLM

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


def test_default_attention_mask_is_causal_when_omitted(tiny_config):
    model = MiniLmForCausalLM(tiny_config)

    mask = model.model._prepare_autoregressive_attention_mask(
        attention_mask=None,
        batch_size=2,
        query_length=3,
        key_length=3,
        past_seen_tokens=0,
        device=torch.device("cpu"),
    )

    expected = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    expected = expected.view(1, 1, 3, 3).expand(2, 1, 3, 3)
    assert torch.equal(mask, expected)


@torch.no_grad()
def test_omitted_attention_mask_hides_future_tokens(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    changed_future = input_ids.clone()
    changed_future[:, -1] = 11

    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
    changed_logits = model(input_ids=changed_future, use_cache=False).logits[:, :-1, :]

    assert torch.allclose(logits, changed_logits, atol=1e-6, rtol=1e-6)
