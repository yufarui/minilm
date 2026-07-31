"""Regression: object-shaped assistant.tool_calls must become a list, not be dropped."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.dataset.sft_dataset import SFTDataset

from .dataset_test_utils import load_local_tokenizer, write_jsonl

PROJECT_ROOT = Path(__file__).parents[2]
SFT_CONV_PATH = PROJECT_ROOT / "src" / "preprocess" / "sft_conversation.py"


def _load_sft_conversation():
    """Avoid importing src.preprocess package (pulls matplotlib via __init__)."""
    spec = importlib.util.spec_from_file_location("sft_conversation_under_test", SFT_CONV_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sft_conv():
    return _load_sft_conversation()


def test_normalize_wraps_object_tool_calls(sft_conv) -> None:
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": '{"name":"search","arguments":{"q":"1"}}',
        },
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]
    out, n = sft_conv.normalize_messages_tool_calls(msgs)
    assert n == 1
    assert isinstance(out[1]["tool_calls"], list)
    assert out[1]["tool_calls"][0]["name"] == "search"
    ok, reason = sft_conv.validate_role_chain(out)
    assert ok, reason


def test_sft_dataset_keeps_object_tool_calls(tmp_path) -> None:
    tok = load_local_tokenizer()
    row = {
        "conversations": [
            {
                "role": "system",
                "content": "sys",
                "tools": [{"name": "search", "description": "d", "parameters": {}}],
            },
            {"role": "user", "content": "find x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": {"name": "search", "arguments": {"q": "x"}},
            },
            {"role": "tool", "content": "hit"},
            {"role": "assistant", "content": "found it"},
        ]
    }
    path = tmp_path / "object_tool_calls.jsonl"
    write_jsonl(path, [row])

    # Disable random system injection for determinism
    ds = SFTDataset(path, tok, pack_bin_size=4096)
    ds.add_system_ratio = 0.0
    sample = next(iter(ds))
    text = tok.decode(sample["input_ids"].tolist())
    assert "<tool_call>" in text
    assert "search" in text
    assert (sample["labels"] != -100).any().item()


def test_tool_calls_fill_returns_false_on_garbage() -> None:
    conv = [{"role": "assistant", "content": "", "tool_calls": "not-json{"}]
    assert SFTDataset._tool_calls_fill(conv) is False
