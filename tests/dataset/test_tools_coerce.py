"""Regression: object-shaped system.tools must become a list for chat_template."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset.dpo_dataset import DPODataset
from src.dataset.sft_dataset import SFTDataset
from src.tokenizer.collect_tokenizer_corpus import _parse_tools_from_system

from .dataset_test_utils import load_local_tokenizer, write_jsonl

TOOLS_OBJECT_JSONL = Path(__file__).parents[1] / "tmp" / "dataset" / "sft_tools_object_sample.jsonl"

SINGLE_TOOL = {
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([SINGLE_TOOL], [SINGLE_TOOL]),
        (SINGLE_TOOL, [SINGLE_TOOL]),
        (json.dumps(SINGLE_TOOL, ensure_ascii=False), [SINGLE_TOOL]),
        (json.dumps([SINGLE_TOOL], ensure_ascii=False), [SINGLE_TOOL]),
        ("", None),
        ("{}", None),
        ({}, None),
        ("not-json", None),
        (123, None),
    ],
)
def test_coerce_tools_shapes(raw, expected):
    assert SFTDataset._coerce_tools(raw) == expected
    assert DPODataset._coerce_tools(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([SINGLE_TOOL], [SINGLE_TOOL]),
        (SINGLE_TOOL, [SINGLE_TOOL]),
        (json.dumps(SINGLE_TOOL, ensure_ascii=False), [SINGLE_TOOL]),
        ("{}", None),
        ({}, None),
        (None, None),
    ],
)
def test_collect_tokenizer_parse_tools(raw, expected):
    msg = {"role": "system", "content": "x", "tools": raw}
    if raw is None:
        msg = {"role": "system", "content": "x"}
    assert _parse_tools_from_system(msg) == expected


def test_sft_object_tools_render_schema_not_dict_keys():
    tok = load_local_tokenizer()
    write_jsonl(
        TOOLS_OBJECT_JSONL,
        [
            {
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are helpful.",
                        # Native JSON object (also covers stringified object via coerce).
                        "tools": SINGLE_TOOL,
                    },
                    {"role": "user", "content": "Weather in Hangzhou?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"name": "get_weather", "arguments": {"city": "Hangzhou"}}
                        ],
                    },
                    {"role": "tool", "content": '{"temp":20}'},
                    {"role": "assistant", "content": "It is 20C in Hangzhou."},
                ]
            }
        ],
    )
    # Deterministic: avoid random system injection path.
    ds = SFTDataset(TOOLS_OBJECT_JSONL, tok, pack_bin_size=4096)
    ds.add_system_ratio = 0.0
    encoded = next(iter(ds))
    text = tok.decode(encoded["input_ids"].tolist())

    assert "get_weather" in text
    # Full schema must appear; bare-object tools would only emit dict keys.
    assert '"name": "get_weather"' in text or '"name":"get_weather"' in text
    assert "parameters" in text
    assert '"city"' in text or "'city'" in text
    # Instructional copy contains an empty <tools></tools> mention; use the last block.
    tools_block = text.rsplit("<tools>", 1)[1].split("</tools>", 1)[0]
    assert "get_weather" in tools_block
    assert "name" in tools_block and "parameters" in tools_block


def test_sft_stringified_object_tools_same_as_list():
    tok = load_local_tokenizer()
    path = TOOLS_OBJECT_JSONL.with_name("sft_tools_string_object_sample.jsonl")
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are helpful.",
                        "tools": json.dumps(SINGLE_TOOL, ensure_ascii=False),
                    },
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ],
    )
    ds = SFTDataset(path, tok, pack_bin_size=4096)
    ds.add_system_ratio = 0.0
    text = tok.decode(next(iter(ds))["input_ids"].tolist())
    tools_block = text.rsplit("<tools>", 1)[1].split("</tools>", 1)[0]
    assert "get_weather" in tools_block
    assert "parameters" in tools_block
