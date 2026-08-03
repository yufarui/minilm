"""DPO chat rows with tool_calls must stay string completions (not char lists)."""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset.dpo_dataset import DPODataset
from tests.dataset.dataset_test_utils import load_local_tokenizer, write_jsonl


def test_dpo_chat_tool_calls_keep_string_completions_and_tool_markup(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "dpo_tool_calls.jsonl"
    write_jsonl(
        path,
        [
            {
                "chosen": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"SF"}',
                                }
                            }
                        ],
                    },
                ],
                "rejected": [
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": "no idea"},
                ],
            }
        ],
    )

    ds = DPODataset(path, tokenizer=tok).as_hf_dataset()
    row = ds[0]

    assert str(ds.features["chosen"]) == "Value('string')"
    assert str(ds.features["rejected"]) == "Value('string')"
    assert isinstance(row["chosen"], str)
    assert isinstance(row["rejected"], str)

    # Null content must not become the literal "None" / character list.
    assert row["chosen"] != "None"
    assert not isinstance(row["chosen"], list)
    assert "<tool_call>" in row["chosen"]
    assert "get_weather" in row["chosen"]
    assert row["chosen"].endswith(tok.eos_token)

    # Prompt must end at the assistant generation boundary for TRL concat.
    assert row["prompt"].endswith("<|im_start|>assistant\n")
    assert not row["prompt"].endswith(row["chosen"])

    # Rejected plain-text completion also keeps eos from the template.
    assert row["rejected"].startswith("no idea")
    assert row["rejected"].endswith(tok.eos_token)


def test_dpo_flat_strings_unchanged(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "dpo_flat.jsonl"
    write_jsonl(
        path,
        [{"prompt": "P", "chosen": "yes", "rejected": "no"}],
    )
    ds = DPODataset(path, tokenizer=tok).as_hf_dataset()
    assert ds[0]["prompt"] == "P"
    assert ds[0]["chosen"] == "yes"
    assert ds[0]["rejected"] == "no"
