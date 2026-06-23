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
def test_causal_lm_default_mask_blocks_future_tokens(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([
        [5, 7, 9, 11, 13, 15],
        [5, 7, 9, 20, 21, 22],
    ])

    logits = model(input_ids=input_ids, use_cache=False).logits

    assert torch.allclose(logits[0, :3], logits[1, :3], atol=1e-6, rtol=1e-6)
