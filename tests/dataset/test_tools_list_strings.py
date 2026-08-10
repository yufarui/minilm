"""Regression: system.tools list-of-JSON-strings must not abort SFT/DPO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.dpo_dataset import DPODataset
from src.dataset.sft_dataset import SFTDataset
from src.preprocess.sft_conversation import normalize_messages_tool_calls
from src.tokenizer.collect_tokenizer_corpus import _parse_tools_from_system
from src.util.tools_normalize import coerce_tools_list_elements

from .dataset_test_utils import load_local_tokenizer, write_jsonl

TMP = Path(__file__).parents[1] / "tmp" / "dataset"

SINGLE_TOOL = {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _tools_block(text: str) -> str:
    # Instructional copy contains an empty <tools></tools> mention; use the last block.
    return text.rsplit("<tools>", 1)[1].split("</tools>", 1)[0]


def test_coerce_tools_list_elements_parses_json_strings_and_drops_nulls():
    raw = [json.dumps(SINGLE_TOOL, ensure_ascii=False), None, "", SINGLE_TOOL]
    assert coerce_tools_list_elements(raw) == [SINGLE_TOOL, SINGLE_TOOL]


def test_coerce_tools_list_elements_skips_non_object_json():
    assert coerce_tools_list_elements(['["not", "a", "schema"]', "not-json"]) == []


def test_parse_tools_from_system_list_of_strings():
    msg = {
        "role": "system",
        "content": "x",
        "tools": [json.dumps(SINGLE_TOOL, ensure_ascii=False)],
    }
    assert _parse_tools_from_system(msg) == [SINGLE_TOOL]


def test_parse_tools_from_system_stringified_list_of_strings():
    msg = {
        "role": "system",
        "content": "x",
        "tools": json.dumps([json.dumps(SINGLE_TOOL, ensure_ascii=False)]),
    }
    assert _parse_tools_from_system(msg) == [SINGLE_TOOL]


def test_preprocess_normalize_repairs_system_tools_list_strings():
    messages = [
        {
            "role": "system",
            "content": "sys",
            "tools": [json.dumps(SINGLE_TOOL, ensure_ascii=False), None],
        },
        {"role": "user", "content": "hi"},
    ]
    out, n = normalize_messages_tool_calls(messages)
    assert n >= 1
    assert out[0]["tools"] == [SINGLE_TOOL]


def test_sft_list_of_json_string_tools_trains_without_crash():
    tok = load_local_tokenizer()
    path = TMP / "sft_tools_list_strings.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are helpful.",
                        "tools": [json.dumps(SINGLE_TOOL, ensure_ascii=False)],
                    },
                    {"role": "user", "content": "Weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"name": "get_weather", "arguments": {"city": "Paris"}}
                        ],
                    },
                    {"role": "tool", "content": '{"temp": 22}'},
                    {"role": "assistant", "content": "It is 22C in Paris."},
                ]
            },
            # A following good row must still be reachable (no mid-iter abort).
            {
                "conversations": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            },
        ],
    )
    ds = SFTDataset(path, tok, pack_bin_size=4096)
    ds.add_system_ratio = 0.0
    rows = list(ds)
    assert len(rows) == 2
    text = tok.decode(rows[0]["input_ids"].tolist())
    block = _tools_block(text)
    assert "get_weather" in block
    assert "parameters" in block


def test_sft_null_tools_element_skips_or_keeps_neighbors():
    tok = load_local_tokenizer()
    path = TMP / "sft_tools_null_elem.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {"role": "system", "content": "sys", "tools": [None]},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "ok"},
                ]
            },
        ],
    )
    ds = SFTDataset(path, tok, pack_bin_size=4096)
    ds.add_system_ratio = 0.0
    rows = list(ds)
    # First row skipped (non-empty tools list, zero valid schemas); second kept.
    assert len(rows) == 1
    text = tok.decode(rows[0]["input_ids"].tolist())
    assert "second" in text
    assert "ok" in text


def test_dpo_list_of_json_string_tools_loads():
    tok = load_local_tokenizer()
    path = TMP / "dpo_tools_list_strings.jsonl"
    tools = [json.dumps(SINGLE_TOOL, ensure_ascii=False)]
    write_jsonl(
        path,
        [
            {
                "chosen": [
                    {"role": "system", "content": "sys", "tools": tools},
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": "sunny"},
                ],
                "rejected": [
                    {"role": "system", "content": "sys", "tools": tools},
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": "no"},
                ],
            }
        ],
    )
    ds = DPODataset(path, tokenizer=tok)
    hf = ds.as_hf_dataset()
    assert len(hf) == 1
    assert "get_weather" in hf[0]["prompt"]
