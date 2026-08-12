"""Non-string content (dict tool results / scalars) must not be wiped by chat_template."""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset.dpo_dataset import DPODataset
from src.dataset.sft_dataset import SFTDataset
from src.preprocess.strategies.sft_pipeline import SftPipelineConfig, SftPreprocessPipeline
from src.preprocess.deduplicate import NearDedupConfig
from src.util.message_content import coerce_content_to_text, normalize_messages_content
from tests.dataset.dataset_test_utils import load_local_tokenizer, write_jsonl


def test_coerce_dict_and_scalars() -> None:
    text, ok = coerce_content_to_text({"temperature": 72, "condition": "sunny"})
    assert ok
    assert json.loads(text) == {"temperature": 72, "condition": "sunny"}

    text, ok = coerce_content_to_text(42)
    assert ok and text == "42"

    text, ok = coerce_content_to_text(True)
    assert ok and text == "true"

    text, ok = coerce_content_to_text(None)
    assert ok and text == ""


def test_normalize_leaves_multipart_lists_untouched() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": "ok"},
    ]
    out, n = normalize_messages_content(messages)
    assert out is not None
    assert n == 0
    assert out[0]["content"] == [{"type": "text", "text": "hi"}]


def test_sft_keeps_dict_tool_observation(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "sft_dict_tool.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {"role": "user", "content": "weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": {"temperature": 72, "condition": "sunny"},
                    },
                    {
                        "role": "assistant",
                        "content": "It is 72F and sunny in Paris.",
                    },
                ]
            }
        ],
    )

    ds = SFTDataset(path, tok, pack_bin_size=8192)
    ds.add_system_ratio = 0.0
    rows = list(ds)
    assert len(rows) == 1
    ids = rows[0]["input_ids"].tolist()
    labels = rows[0]["labels"].tolist()
    text = tok.decode(ids, skip_special_tokens=False)
    assert "<tool_response>" in text
    assert '"temperature": 72' in text or '"temperature":72' in text
    assert "sunny" in text
    supervised = tok.decode(
        [t for t, lab in zip(ids, labels) if lab != -100],
        skip_special_tokens=False,
    )
    assert "It is 72F and sunny in Paris." in supervised


def test_sft_scalar_assistant_content_not_empty_eos(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "sft_scalar.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": 4},
                ]
            }
        ],
    )
    ds = SFTDataset(path, tok, pack_bin_size=8192)
    ds.add_system_ratio = 0.0
    rows = list(ds)
    assert len(rows) == 1
    ids = rows[0]["input_ids"].tolist()
    labels = rows[0]["labels"].tolist()
    supervised = tok.decode(
        [t for t, lab in zip(ids, labels) if lab != -100],
        skip_special_tokens=False,
    )
    assert "4" in supervised
    # Must not be empty-EOS-only supervision.
    assert supervised.strip() != "<|im_end|>"


def test_preprocess_serializes_dict_tool_content(tmp_path: Path) -> None:
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    write_jsonl(
        inp,
        [
            {
                "conversations": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "get_weather", "arguments": {}}],
                    },
                    {"role": "tool", "content": {"ok": True}},
                    {"role": "assistant", "content": "done"},
                ]
            }
        ],
    )
    cfg = SftPipelineConfig(
        filter_refuse_replies=False,
        exact_dedup=False,
        near_dedup=NearDedupConfig(enabled=False),
        allowed_langs=[],
        run_diagnostics=False,
        min_chars=1,
    )
    stats = SftPreprocessPipeline(cfg).run(inp, out)
    assert stats.output_lines == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    tool = next(m for m in row["conversations"] if m.get("role") == "tool")
    assert isinstance(tool["content"], str)
    assert json.loads(tool["content"]) == {"ok": True}


def test_dpo_prefix_keeps_dict_tool_content(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "dpo_dict_tool.jsonl"
    write_jsonl(
        path,
        [
            {
                "chosen": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "get_weather", "arguments": {}}],
                    },
                    {"role": "tool", "content": {"temp": 10}},
                    {"role": "assistant", "content": "cold"},
                ],
                "rejected": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "get_weather", "arguments": {}}],
                    },
                    {"role": "tool", "content": {"temp": 10}},
                    {"role": "assistant", "content": "idk"},
                ],
            }
        ],
    )
    ds = DPODataset(path, tokenizer=tok).as_hf_dataset()
    assert len(ds) == 1
    assert '"temp": 10' in ds[0]["prompt"] or '"temp":10' in ds[0]["prompt"]
    assert ds[0]["chosen"] == "cold"
    assert ds[0]["rejected"] == "idk"
