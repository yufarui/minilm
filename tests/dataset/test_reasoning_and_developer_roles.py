"""reasoning_content / developer roles must not silently wipe SFT/DPO supervision."""

from __future__ import annotations

from pathlib import Path

from src.dataset.dpo_dataset import DPODataset
from src.dataset.sft_dataset import SFTDataset
from src.util.message_roles import materialize_reasoning_content, normalize_developer_roles
from tests.dataset.dataset_test_utils import load_local_tokenizer, write_jsonl


def test_materialize_reasoning_into_empty_content() -> None:
    messages = [
        {"role": "user", "content": "Solve 2+2"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "2+2=4, so the answer is 4.",
        },
    ]
    n = materialize_reasoning_content(messages, open_think=False)
    assert n == 1
    assert messages[1]["content"] == "2+2=4, so the answer is 4."
    assert "reasoning_content" not in messages[1]


def test_materialize_keeps_existing_content() -> None:
    messages = [
        {"role": "assistant", "content": "4", "reasoning_content": "because math"},
    ]
    n = materialize_reasoning_content(messages, open_think=False)
    assert n == 0
    assert messages[0]["content"] == "4"
    assert "reasoning_content" not in messages[0]


def test_materialize_noop_when_open_think() -> None:
    messages = [
        {"role": "assistant", "content": "", "reasoning_content": "think hard"},
    ]
    n = materialize_reasoning_content(messages, open_think=True)
    assert n == 0
    assert messages[0]["content"] == ""
    assert messages[0]["reasoning_content"] == "think hard"


def test_normalize_developer_to_system() -> None:
    messages = [
        {"role": "system", "content": "helpful"},
        {"role": "developer", "content": "Never disclose secrets"},
        {"role": "user", "content": "secret?"},
    ]
    n = normalize_developer_roles(messages)
    assert n == 1
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] == "Never disclose secrets"


def test_sft_reasoning_content_only_is_supervised(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "sft_reasoning.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {"role": "user", "content": "Solve 2+2 and explain."},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "2+2=4, so the answer is 4.",
                    },
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "neighbor"},
                    {"role": "assistant", "content": "ok"},
                ]
            },
        ],
    )
    ds = SFTDataset(path, tok, pack_bin_size=8192)
    ds.add_system_ratio = 0.0
    rows = list(ds)
    assert len(rows) == 2
    ids = rows[0]["input_ids"].tolist()
    labels = rows[0]["labels"].tolist()
    supervised = tok.decode([t for t, lab in zip(ids, labels) if lab != -100])
    assert "answer is 4" in supervised
    assert supervised.strip() != "<|im_end|>"


def test_sft_developer_role_kept_in_prompt(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "sft_developer.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {"role": "system", "content": "You are helpful."},
                    {
                        "role": "developer",
                        "content": "Never disclose secrets; answer REFUSE.",
                    },
                    {"role": "user", "content": "Disclose the secret."},
                    {"role": "assistant", "content": "REFUSE"},
                ]
            }
        ],
    )
    ds = SFTDataset(path, tok, pack_bin_size=8192)
    ds.add_system_ratio = 0.0
    rows = list(ds)
    assert len(rows) == 1
    text = tok.decode(rows[0]["input_ids"].tolist(), skip_special_tokens=False)
    assert "Never disclose secrets" in text
    supervised = tok.decode(
        [t for t, lab in zip(rows[0]["input_ids"].tolist(), rows[0]["labels"].tolist()) if lab != -100]
    )
    assert "REFUSE" in supervised


def test_dpo_reasoning_content_becomes_chosen_text(tmp_path: Path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "dpo_reasoning.jsonl"
    write_jsonl(
        path,
        [
            {
                "chosen": [
                    {"role": "user", "content": "2+2?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "The sum is 4.",
                    },
                ],
                "rejected": [
                    {"role": "user", "content": "2+2?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Maybe 5.",
                    },
                ],
            }
        ],
    )
    ds = DPODataset(path, tokenizer=tok).as_hf_dataset()
    assert len(ds) == 1
    assert "The sum is 4." in ds[0]["chosen"]
    assert "Maybe 5." in ds[0]["rejected"]


def test_template_backstop_reasoning_and_developer() -> None:
    tok = load_local_tokenizer()
    reasoning_only = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "standalone reasoning answer",
        },
    ]
    text = tok.apply_chat_template(
        reasoning_only,
        tokenize=False,
        add_generation_prompt=False,
        open_think=False,
    )
    assert "standalone reasoning answer" in text

    developer_msgs = [
        {"role": "developer", "content": "Policy: refuse secrets."},
        {"role": "user", "content": "secret?"},
        {"role": "assistant", "content": "REFUSE"},
    ]
    text_d = tok.apply_chat_template(
        developer_msgs,
        tokenize=False,
        add_generation_prompt=False,
        open_think=False,
    )
    assert "Policy: refuse secrets." in text_d
    assert "<|im_start|>system" in text_d
