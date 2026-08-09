"""Regression: tool_calls list elements that are JSON strings must not crash SFT."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.dataset.sft_dataset import SFTDataset
from src.preprocess.sft_conversation import normalize_messages_tool_calls
from src.util.tool_calls_normalize import normalize_tool_calls_list
from tests.dataset.dataset_test_utils import load_local_tokenizer


def test_normalize_tool_calls_list_parses_json_string_elements() -> None:
    raw = ['{"name":"get_weather","arguments":{"city":"Paris"}}']
    out = normalize_tool_calls_list(raw)
    assert out == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_normalize_tool_calls_list_parses_stringified_function() -> None:
    raw = [
        {
            "type": "function",
            "function": '{"name":"get_weather","arguments":{"city":"Paris"}}',
        }
    ]
    out = normalize_tool_calls_list(raw)
    assert out is not None
    assert out[0]["function"]["name"] == "get_weather"
    assert out[0]["function"]["arguments"] == {"city": "Paris"}


def test_normalize_tool_calls_list_rejects_null_element() -> None:
    assert normalize_tool_calls_list([None]) is None


def test_preprocess_normalize_repairs_list_of_json_strings() -> None:
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": ['{"name":"get_weather","arguments":{"city":"Paris"}}'],
        },
    ]
    out, n = normalize_messages_tool_calls(messages)
    assert n == 1
    assert out[1]["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Paris"}}
    ]


def test_sft_encodes_list_of_json_string_tool_calls() -> None:
    tok = load_local_tokenizer()
    ds = SFTDataset.__new__(SFTDataset)
    ds.tokenizer = tok
    ds.max_seq_len = 8192
    ds.add_system_ratio = 0.0
    import re

    eos = getattr(tok, "eos_token", None) or "<|im_end|>"
    ds._assistant_block = re.compile(
        rf"<\|im_start\|>assistant\n(.*?){re.escape(eos)}",
        re.DOTALL,
    )

    conv = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": ['{"name":"get_weather","arguments":{"city":"Paris"}}'],
        },
    ]
    enc = ds._encode_conversation(conv)
    assert enc is not None
    input_ids, labels = enc
    text = tok.decode(input_ids)
    assert "<tool_call>" in text
    assert "get_weather" in text
    assert "Paris" in text
    assert any(x != -100 for x in labels)


def test_sft_iter_skips_null_tool_call_element_without_crash() -> None:
    tok = load_local_tokenizer()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rows.jsonl"
        rows = [
            {
                "conversations": [
                    {"role": "user", "content": "ok?"},
                    {"role": "assistant", "content": "yes"},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [None],
                    },
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "ok2?"},
                    {"role": "assistant", "content": "yes2"},
                ]
            },
        ]
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8",
        )
        ds = SFTDataset(path, tok, pack_bin_size=8192)
        ds.add_system_ratio = 0.0
        emitted = list(ds)
        assert len(emitted) == 2
