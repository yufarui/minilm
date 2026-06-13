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
def test_forward_without_attention_mask_is_causal(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[5, 6, 7, 8]])
    changed_future = torch.tensor([[5, 9, 10, 11]])

    logits = model(input_ids=input_ids, use_cache=False).logits
    changed_logits = model(input_ids=changed_future, use_cache=False).logits

    torch.testing.assert_close(logits[:, 0, :], changed_logits[:, 0, :], rtol=0, atol=0)
