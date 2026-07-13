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


@torch.no_grad()
def test_causal_lm_default_attention_is_causal(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    input_a = torch.cat([prefix, torch.tensor([[5, 6]], dtype=torch.long)], dim=1)
    input_b = torch.cat([prefix, torch.tensor([[7, 8]], dtype=torch.long)], dim=1)

    logits_a = model(input_ids=input_a, use_cache=False).logits
    logits_b = model(input_ids=input_b, use_cache=False).logits

    assert torch.allclose(
        logits_a[:, : prefix.shape[1], :],
        logits_b[:, : prefix.shape[1], :],
        atol=1e-6,
    )
