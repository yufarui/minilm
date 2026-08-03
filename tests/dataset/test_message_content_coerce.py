"""OpenAI multipart message content must become plain text before chat_template."""

from __future__ import annotations

from pathlib import Path

from src.dataset.sft_dataset import SFTDataset
from src.util.message_content import coerce_content_to_text, normalize_messages_content
from tests.dataset.dataset_test_utils import load_local_tokenizer, write_jsonl


def test_coerce_openai_multipart_text_parts() -> None:
    text, ok = coerce_content_to_text(
        [{"type": "text", "text": "What is 2+2?"}, {"type": "text", "text": "Explain."}]
    )
    assert ok
    assert text == "What is 2+2?\nExplain."


def test_coerce_skips_image_parts() -> None:
    text, ok = coerce_content_to_text(
        [
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
        ]
    )
    assert ok
    assert text == "caption"


def test_normalize_messages_content_rewrites_lists() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    out, n = normalize_messages_content(messages)
    assert out is not None
    assert n == 2
    assert out[0]["content"] == "hi"
    assert out[1]["content"] == "hello"


def test_sft_multipart_content_keeps_supervision(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "sft_multipart.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What is 2+2?"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "4"}],
                    },
                ]
            }
        ],
    )

    ds = SFTDataset(path, tok, pack_bin_size=8192)
    # Disable random system insert for stable assertion.
    ds.add_system_ratio = 0.0
    rows = list(ds)
    assert len(rows) == 1
    ids = rows[0]["input_ids"].tolist()
    labels = rows[0]["labels"].tolist()
    text = tok.decode(ids, skip_special_tokens=False)
    assert "What is 2+2?" in text
    assert "4" in text
    # Must supervise the assistant answer, not an empty turn.
    supervised = [t for t, lab in zip(ids, labels) if lab != -100]
    assert supervised
    assert "4" in tok.decode(supervised, skip_special_tokens=False)
