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


def test_missing_attention_mask_defaults_to_causal_mask(tiny_config):
    mask = MiniLMModel._prepare_autoregressive_attention_mask(
        attention_mask=None,
        batch_size=2,
        query_length=4,
        key_length=4,
        past_seen_tokens=0,
        device=torch.device("cpu"),
    )
    expected = torch.tril(torch.ones((1, 1, 4, 4), dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_missing_attention_mask_with_cache_allows_past_only(tiny_config):
    mask = MiniLMModel._prepare_autoregressive_attention_mask(
        attention_mask=None,
        batch_size=2,
        query_length=2,
        key_length=5,
        past_seen_tokens=3,
        device=torch.device("cpu"),
    )
    expected = torch.tensor(
        [[[[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]]],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


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
