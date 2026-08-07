"""Regression: blank tool_calls.arguments must not render invalid JSON into SFT labels."""

from __future__ import annotations

import json

from tests.dataset.dataset_test_utils import load_local_tokenizer


def _render_assistant_tool_call(arguments) -> str:
    tok = load_local_tokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "search", "arguments": arguments}],
        },
    ]
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        open_think=False,
    )


def test_empty_string_tool_call_arguments_render_as_empty_object() -> None:
    text = _render_assistant_tool_call("")
    assert '"arguments": }' not in text
    assert '"arguments": {}' in text
    # Extract the tool_call JSON object and ensure it parses.
    start = text.index("<tool_call>\n") + len("<tool_call>\n")
    end = text.index("\n</tool_call>", start)
    payload = json.loads(text[start:end])
    assert payload == {"name": "search", "arguments": {}}


def test_whitespace_tool_call_arguments_render_as_empty_object() -> None:
    text = _render_assistant_tool_call("   \n\t")
    assert '"arguments": {}' in text


def test_nonempty_string_arguments_still_emitted_raw() -> None:
    text = _render_assistant_tool_call('{"q": "weather"}')
    assert '"arguments": {"q": "weather"}' in text


def test_sft_labels_supervise_valid_json_for_empty_arguments() -> None:
    from src.dataset.sft_dataset import SFTDataset

    tok = load_local_tokenizer()
    ds = SFTDataset.__new__(SFTDataset)
    ds.tokenizer = tok
    ds.add_system_ratio = 0.0
    ds.max_seq_len = 8192
    import re

    eos = tok.eos_token or "<|im_end|>"
    ds._assistant_block = re.compile(
        rf"<\|im_start\|>assistant\n(.*?){re.escape(eos)}",
        re.DOTALL,
    )

    conv = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "search", "arguments": ""}],
        },
    ]
    encoded = SFTDataset._encode_conversation(ds, conv)
    assert encoded is not None
    input_ids, labels = encoded
    supervised = tok.decode([t for t, lab in zip(input_ids, labels) if lab != -100])
    assert '"arguments": }' not in supervised
    assert '"arguments": {}' in supervised
