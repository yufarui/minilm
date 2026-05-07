import torch

from src.model.model import MiniLmForCausalLM

@torch.no_grad()
def test_incremental_decode_matches_full_logits_close(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 6))
    seqlen = input_ids.shape[1]
    causal_mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, dtype=torch.long))

    full = model(input_ids=input_ids, attention_mask=causal_mask, use_cache=False).logits[:, -1, :]

    first = model(input_ids=input_ids[:, :5], attention_mask=causal_mask[:, :, :5, :5], use_cache=True)
    step = model(
        input_ids=input_ids[:, 5:],
        attention_mask=causal_mask[:, :, 5:, :],
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    inc = step.logits[:, -1, :]

    assert torch.allclose(full, inc, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_incremental_decode_without_attention_mask_is_causal(tiny_config):
    model = MiniLmForCausalLM(tiny_config).eval()
    input_ids = torch.randint(1, tiny_config.vocab_size, (1, 6))

    full = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]

    first = model(input_ids=input_ids[:, :5], use_cache=True)
    step = model(
        input_ids=input_ids[:, 5:],
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    inc = step.logits[:, -1, :]

    assert torch.allclose(full, inc, atol=1e-4, rtol=1e-4)
