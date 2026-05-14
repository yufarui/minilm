import pytest
import torch

from src.model.model import MiniLMModel

def test_minilm_model_forward_shape_and_cache(tiny_config):
    model = MiniLMModel(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    out = model(input_ids=input_ids, use_cache=True)
    assert out.last_hidden_state.shape == (2, 6, tiny_config.hidden_size)
    assert out.past_key_values is not None
    assert torch.isfinite(out.last_hidden_state).all()


@torch.no_grad()
def test_minilm_model_default_attention_mask_is_causal(tiny_config):
    model = MiniLMModel(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 6))
    seqlen = input_ids.shape[1]
    causal_mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, dtype=torch.bool))

    default_out = model(input_ids=input_ids, use_cache=False).last_hidden_state
    explicit_out = model(
        input_ids=input_ids,
        attention_mask=causal_mask,
        use_cache=False,
    ).last_hidden_state

    assert torch.allclose(default_out, explicit_out, atol=1e-5, rtol=1e-5)


def test_minilm_model_inputs_embeds_path(tiny_config):
    model = MiniLMModel(tiny_config).eval()
    embeds = torch.randn(2, 5, tiny_config.hidden_size)
    out = model(inputs_embeds=embeds, use_cache=False)
    assert out.last_hidden_state.shape == embeds.shape


def test_minilm_model_invalid_input_raises(tiny_config):
    model = MiniLMModel(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 3))
    embeds = torch.randn(1, 3, tiny_config.hidden_size)
    with pytest.raises(ValueError):
        model(input_ids=input_ids, inputs_embeds=embeds)
