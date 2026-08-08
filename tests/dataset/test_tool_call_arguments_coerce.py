"""Regression: non-JSON tool_calls.arguments must not poison SFT tool supervision."""

from __future__ import annotations

import json
import re

from src.dataset.sft_dataset import SFTDataset
from src.preprocess.sft_conversation import normalize_messages_tool_calls
from src.util.tool_call_arguments import coerce_tool_call_arguments
from tests.dataset.dataset_test_utils import load_local_tokenizer


def _tool_call_payload(text: str) -> dict:
    start = text.index("<tool_call>\n") + len("<tool_call>\n")
    end = text.index("\n</tool_call>", start)
    return json.loads(text[start:end])


def _make_sft_dataset(tok) -> SFTDataset:
    ds = SFTDataset.__new__(SFTDataset)
    ds.tokenizer = tok
    ds.add_system_ratio = 0.0
    ds.max_seq_len = 8192
    eos = tok.eos_token or "<|im_end|>"
    ds._assistant_block = re.compile(
        rf"<\|im_start\|>assistant\n(.*?){re.escape(eos)}",
        re.DOTALL,
    )
    return ds


def test_coerce_python_repr_arguments_to_object() -> None:
    assert coerce_tool_call_arguments("{'city': 'Paris'}") == {"city": "Paris"}


def test_coerce_bare_string_arguments_to_empty_object() -> None:
    assert coerce_tool_call_arguments("hello world") == {}


def test_coerce_valid_json_string_arguments_to_object() -> None:
    assert coerce_tool_call_arguments('{"q": "weather"}') == {"q": "weather"}


def test_coerce_empty_string_arguments_to_empty_object() -> None:
    assert coerce_tool_call_arguments("") == {}
    assert coerce_tool_call_arguments("  \n") == {}


def test_sft_encode_python_repr_arguments_supervises_valid_json() -> None:
    tok = load_local_tokenizer()
    ds = _make_sft_dataset(tok)
    conv = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_weather", "arguments": "{'city': 'Paris'}"}],
        },
    ]
    encoded = SFTDataset._encode_conversation(ds, conv)
    assert encoded is not None
    input_ids, labels = encoded
    supervised = tok.decode([t for t, lab in zip(input_ids, labels) if lab != -100])
    assert "{'city': 'Paris'}" not in supervised
    assert _tool_call_payload(supervised) == {
        "name": "get_weather",
        "arguments": {"city": "Paris"},
    }


def test_sft_encode_openai_nested_python_repr_arguments() -> None:
    tok = load_local_tokenizer()
    ds = _make_sft_dataset(tok)
    conv = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search", "arguments": "{'q': 1}"}}
            ],
        },
    ]
    encoded = SFTDataset._encode_conversation(ds, conv)
    assert encoded is not None
    input_ids, labels = encoded
    supervised = tok.decode([t for t, lab in zip(input_ids, labels) if lab != -100])
    assert _tool_call_payload(supervised) == {
        "name": "search",
        "arguments": {"q": 1},
    }


def test_sft_encode_keeps_valid_json_string_arguments() -> None:
    tok = load_local_tokenizer()
    ds = _make_sft_dataset(tok)
    conv = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "search", "arguments": '{"q": "weather"}'}],
        },
    ]
    encoded = SFTDataset._encode_conversation(ds, conv)
    assert encoded is not None
    input_ids, labels = encoded
    supervised = tok.decode([t for t, lab in zip(input_ids, labels) if lab != -100])
    assert _tool_call_payload(supervised) == {
        "name": "search",
        "arguments": {"q": "weather"},
    }


def test_preprocess_normalizes_python_repr_arguments() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "search", "arguments": "{'q': 'x'}"}],
        },
    ]
    out, repaired = normalize_messages_tool_calls(messages)
    assert repaired >= 1
    assert out[1]["tool_calls"][0]["arguments"] == {"q": "x"}
