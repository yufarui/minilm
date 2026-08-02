"""回归：OpenAI legacy role=function 与残缺 tool_calls.arguments 不得静默丢数据 / 崩训练。"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from src.dataset.sft_dataset import SFTDataset
from src.preprocess.sft_conversation import normalize_legacy_tool_roles, validate_role_chain
from tests.dataset.dataset_test_utils import write_jsonl

PROJECT_ROOT = Path(__file__).parents[2]
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer" / "minilm"
TMP_DIR = Path(__file__).parents[1] / "tmp" / "dataset"


def _tokenizer():
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)


def test_chat_template_renders_legacy_function_role_as_tool_response():
    tok = _tokenizer()
    tools = [{"name": "search", "parameters": {"type": "object"}}]
    msgs = [
        {"role": "system", "content": "You are helpful.", "tools": tools},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "search", "arguments": {"q": "weather"}}],
        },
        {"role": "function", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, tools=tools, open_think=False
    )
    assert "<tool_response>" in text
    assert "sunny" in text
    assert text.count("<|im_start|>assistant") == 2


def test_chat_template_missing_tool_call_arguments_defaults_to_empty_object():
    tok = _tokenizer()
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "search"}]},
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, open_think=False
    )
    assert '"arguments": {}' in text
    assert "<tool_call>" in text


def test_chat_template_nested_function_missing_arguments_defaults():
    tok = _tokenizer()
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search"}}],
        },
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, open_think=False
    )
    assert '"name": "search"' in text
    assert '"arguments": {}' in text


def test_normalize_legacy_tool_roles_and_validate_chain():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "search", "arguments": {}}],
        },
        {"role": "function", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    ok_before, reason = validate_role_chain(msgs)
    assert ok_before, reason
    norm, n = normalize_legacy_tool_roles(msgs)
    assert n == 1
    assert norm[2]["role"] == "tool"
    ok_after, reason2 = validate_role_chain(norm)
    assert ok_after, reason2


def test_sft_dataset_encodes_function_role_and_incomplete_tool_calls():
    tok = _tokenizer()
    path = TMP_DIR / "sft_function_role_and_args.jsonl"
    rows = [
        {
            "conversations": [
                {
                    "role": "system",
                    "content": "sys",
                    "tools": [{"name": "search", "parameters": {"type": "object"}}],
                },
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": "search"}],  # 缺 arguments
                },
                {"role": "function", "content": "sunny"},
                {"role": "assistant", "content": "It is sunny."},
            ]
        }
    ]
    write_jsonl(path, rows)
    ds = SFTDataset(path, tok, pack_bin_size=4096)
    encoded = next(iter(ds))
    text = tok.decode(encoded["input_ids"].tolist())
    assert "<tool_response>" in text or "sunny" in text
    assert "It is sunny" in text
    assert any(x != -100 for x in encoded["labels"].tolist())
