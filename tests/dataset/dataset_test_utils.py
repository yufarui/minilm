import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

TESTS_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = Path(__file__).parents[2]
PREPROCESS_TMP_ROOT = TESTS_ROOT / "tmp" / "preprocess"
PRETRAIN_JSONL = PREPROCESS_TMP_ROOT / "pretrain_pipeline" / "pretrain_output.jsonl"
SFT_JSONL = PREPROCESS_TMP_ROOT / "sft_pipeline" / "sft_output.jsonl"
DPO_JSONL = TESTS_ROOT / "tmp" / "dataset" / "dpo_mock.jsonl"
PRETRAIN_SCHEDULE_JSONL = TESTS_ROOT / "tmp" / "dataset" / "pretrain_schedule_input.jsonl"
# SFTDataset：system.tools / assistant.tool_calls 为 JSON 字符串的样例（由 test_dataset_loading 写入）
SFT_TOOLS_STRING_JSONL = TESTS_ROOT / "tmp" / "dataset" / "sft_tools_string_sample.jsonl"
LOCAL_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer" / "minilm"


def load_local_tokenizer():
    if not LOCAL_TOKENIZER_DIR.exists():
        pytest.skip(f"本地 tokenizer 不存在：{LOCAL_TOKENIZER_DIR}")
    tok = AutoTokenizer.from_pretrained(str(LOCAL_TOKENIZER_DIR), trust_remote_code=True)
    original_apply_chat_template = tok.apply_chat_template

    def _apply_chat_template_adapted(*args, **kwargs):
        kwargs.setdefault("tokenize", True)
        out = original_apply_chat_template(*args, **kwargs)
        if hasattr(out, "get") and out.get("input_ids") is not None:
            return out["input_ids"]
        if isinstance(out, str):
            return tok.encode(out, add_special_tokens=False)
        return out

    tok.apply_chat_template = _apply_chat_template_adapted  # type: ignore[method-assign]
    return tok


def ensure_preprocess_tmp() -> None:
    missing = [str(p) for p in (PRETRAIN_JSONL, SFT_JSONL) if not p.exists()]
    if missing:
        pytest.fail(
            "预处理测试产物不存在，请先运行 "
            "`pytest tests/preprocess/test_preprocess_pipelines.py` 生成："
            + ", ".join(missing)
        )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
