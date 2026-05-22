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
def test_minilm_model_defaults_to_causal_attention(tiny_config):
    model = MiniLMModel(tiny_config).eval()
    input_ids = torch.randint(1, tiny_config.vocab_size, (1, 6))
    explicit_mask = torch.tril(torch.ones(1, 1, 6, 6, dtype=torch.bool))

    default = model(input_ids=input_ids).last_hidden_state
    explicit = model(input_ids=input_ids, attention_mask=explicit_mask).last_hidden_state

    assert torch.allclose(default, explicit, atol=1e-5, rtol=1e-5)


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
