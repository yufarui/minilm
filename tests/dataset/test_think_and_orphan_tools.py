"""Regressions for silent SFT target corruption via think tags / orphan tools."""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset.sft_dataset import SFTDataset
from src.preprocess.strategies.sft_pipeline import SftPipelineConfig, SftPreprocessPipeline
from src.ref_model.tokenizer_local import get_auto_tokenizer_local
from src.util.path_util import resolve_under_project
from src.util.tool_chain import has_orphan_tool_messages


def _tokenizer():
    return get_auto_tokenizer_local(
        resolve_under_project("tokenizer/minilm"), trust_remote_code=True
    )


def test_chat_template_keeps_assistant_text_with_lone_think_close_tag() -> None:
    tok = _tokenizer()
    messages = [
        {"role": "user", "content": "What does </think> mean?"},
        {
            "role": "assistant",
            "content": "The marker </think> closes a think block. Full answer follows.",
        },
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, open_think=False
    )
    assert "The marker </think> closes a think block. Full answer follows." in text
    assert "<|im_start|>assistant\n closes a think block" not in text


def test_chat_template_still_strips_balanced_think_block() -> None:
    tok = _tokenizer()
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "<think>secret</think>\nVisible answer.",
        },
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, open_think=False
    )
    assert "Visible answer." in text
    assert "secret" not in text.split("assistant\n", 1)[-1].split("<|im_end|>", 1)[0]


def test_has_orphan_tool_messages_detects_missing_tool_calls() -> None:
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": '{"temp":72}'},
        {"role": "assistant", "content": "72F"},
    ]
    assert has_orphan_tool_messages(msgs) is True


def test_has_orphan_tool_messages_allows_valid_tool_chain() -> None:
    msgs = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_weather", "arguments": {}}],
        },
        {"role": "tool", "content": '{"temp":72}'},
        {"role": "assistant", "content": "72F"},
    ]
    assert has_orphan_tool_messages(msgs) is False


def test_sft_dataset_skips_orphan_tool_rows(tmp_path: Path) -> None:
    tok = _tokenizer()
    path = tmp_path / "orphan.jsonl"
    row = {
        "conversations": [
            {"role": "user", "content": "weather in SF today please?"},
            {"role": "assistant", "content": "", "tool_calls": "NOT_JSON{{"},
            {"role": "tool", "content": '{"temp":72}'},
            {"role": "assistant", "content": "It is seventy two degrees outside today."},
        ]
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    ds = SFTDataset(path, tok, pack_bin_size=8192)
    assert list(ds) == []


def test_sft_preprocess_drops_orphan_tools_when_strict_role_order_false(
    tmp_path: Path,
) -> None:
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {
            "id": 1,
            "conversations": [
                {"role": "user", "content": "weather in SF today please tell me?"},
                {"role": "assistant", "content": "", "tool_calls": "NOT_JSON{{"},
                {"role": "tool", "content": '{"temp":72}'},
                {
                    "role": "assistant",
                    "content": "It is seventy two degrees outside today.",
                },
            ],
        },
        {
            "id": 2,
            "conversations": [
                {"role": "user", "content": "hello there friend how are you doing?"},
                {
                    "role": "assistant",
                    "content": "I am doing well thank you for asking today.",
                },
            ],
        },
    ]
    with inp.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg = SftPipelineConfig(
        strict_role_order=False,
        repair_tool_calls=True,
        filter_refuse_replies=False,
        drop_think_samples=False,
        min_chars=20,
        allowed_langs=[],
        exact_dedup=False,
        run_diagnostics=False,
    )
    cfg.near_dedup.enabled = False
    stats = SftPreprocessPipeline(cfg).run(inp, out)
    kept = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert stats.skipped_role_order == 1
    assert "orphan_tool_messages" in stats.role_violation_examples
    assert len(kept) == 1
    assert kept[0]["id"] == 2
