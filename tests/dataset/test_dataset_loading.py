import json
import pytest
from itertools import islice
from pathlib import Path

from src.dataset.dpo_dataset import DPODataset
from src.dataset import pre_train_dataset as pretrain_module
from src.dataset.pre_train_dataset import PreTrainDataset
from src.dataset.sft_dataset import SFTDataset

from .dataset_test_utils import (
    DPO_JSONL,
    PRETRAIN_JSONL,
    PRETRAIN_SCHEDULE_JSONL,
    SFT_JSONL,
    SFT_TOOLS_STRING_JSONL,
    ensure_preprocess_tmp,
    load_local_tokenizer,
    write_jsonl,
)


def test_pretrain_dataset_load_from_preprocess_tmp() -> None:
    ensure_preprocess_tmp()
    tok = load_local_tokenizer()
    ds = PreTrainDataset(PRETRAIN_JSONL, tok, pack_bin_size=64)
    sample = next(iter(ds))
    assert "input_ids" in sample and "labels" in sample
    assert sample["input_ids"].shape == sample["labels"].shape


def test_pretrain_dataset_pack_bin_schedule() -> None:
    tok = load_local_tokenizer()
    if not PRETRAIN_SCHEDULE_JSONL.exists():
        pytest.fail(
            "缺少 pack_bin_schedule 测试数据，请先运行 "
            "`pytest tests/dataset_test/test_dataset_data_building.py`"
        )

    ds = PreTrainDataset(
        PRETRAIN_SCHEDULE_JSONL,
        tok,
        pack_bin_size=64,
        pack_bin_schedule=[
            {"until_index": 2, "pack_bin_size": 8},
            {"pack_bin_size": 16},
        ],
    )
    assert ds.pack_bin_size == 16

    first_three = list(islice(iter(ds), 3))
    assert len(first_three) == 3

    # 前两条由第一阶段产出，应严格按 8 token 切分。
    assert first_three[0]["input_ids"].shape[0] == 8
    assert first_three[1]["input_ids"].shape[0] == 8

    # 后续阶段块长为 16，最后一条允许短于 16。
    assert 0 < first_three[2]["input_ids"].shape[0] <= 16


def test_sft_dataset_load_from_preprocess_tmp() -> None:
    ensure_preprocess_tmp()
    tok = load_local_tokenizer()
    ds = SFTDataset(SFT_JSONL, tok, pack_bin_size=128)

    # 预处理样本中应包含至少一条 tool 消息（由 test_preprocess_pipelines 构造）。
    has_tool_row = False
    with open(SFT_JSONL, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if any(m.get("role") == "tool" for m in row.get("conversations", [])):
                has_tool_row = True
                break
    assert has_tool_row

    # 打包后应能在解码文本中看到 tool 相关内容，证明 tool 路径被实际编码。
    decoded_texts = [tok.decode(item["input_ids"].tolist()) for item in islice(iter(ds), 32)]
    assert any(("tool" in txt) and ("result" in txt) for txt in decoded_texts)

    sample = next(iter(ds))

    print("sample\n", sample)
    print("decode\n", tok.decode(sample["input_ids"]))

    assert "input_ids" in sample and "labels" in sample
    assert sample["input_ids"].shape == sample["labels"].shape
    assert (sample["labels"] != -100).any().item(), "assistant 段应对齐到非 -100 的监督位"


def test_dpo_dataset_load_with_mock_jsonl() -> None:
    if not DPO_JSONL.exists():
        pytest.fail(
            "缺少 DPO mock 数据，请先运行 "
            "`pytest tests/dataset_test/test_dataset_data_building.py`"
        )
    tok = load_local_tokenizer()
    ds = DPODataset(DPO_JSONL, tokenizer=tok).as_hf_dataset()
    assert len(ds) == 10
    row = ds[0]
    assert row["prompt"]
    assert row["prompt"].endswith("<|im_start|>assistant\n")
    assert row["chosen"]
    assert row["rejected"]


def test_sft_dataset_loads_stringified_tools_and_tool_calls() -> None:
    tok = load_local_tokenizer()
    # 需足够大：system tools 定义较长，过小会在 tool 回复前被截断，解码中看不到 "69"。
    ds = SFTDataset(SFT_TOOLS_STRING_JSONL, tok, pack_bin_size=4096)
    encoded = next(iter(ds))
    print("encoded\n", encoded)
    text = tok.decode(encoded["input_ids"].tolist())
    print("text\n", text)
    # tools/tool_calls 为字符串 JSON 时也应可被解析并编码进模板文本。
    assert ("random_number" in text) or ("get_exchange_rate" in text)
    assert "69" in text


def test_sft_tool_calls_fill_rejects_invalid_non_empty_json() -> None:
    conv = [{"role": "assistant", "content": "", "tool_calls": "{not-json"}]
    assert SFTDataset._tool_calls_fill(conv) is False


def test_sft_dataset_skips_invalid_tool_calls_json(tmp_path) -> None:
    tok = load_local_tokenizer()
    path = tmp_path / "sft_invalid_tool_calls.jsonl"
    write_jsonl(
        path,
        [
            {
                "conversations": [
                    {"role": "user", "content": "call a tool"},
                    {"role": "assistant", "content": "", "tool_calls": "{not-json"},
                    {"role": "tool", "content": "bad result"},
                    {"role": "assistant", "content": "bad sample should be skipped"},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "valid answer"},
                ]
            },
        ],
    )

    ds = SFTDataset(path, tok, pack_bin_size=512)
    encoded = next(iter(ds))
    text = tok.decode(encoded["input_ids"].tolist())
    assert "valid answer" in text
    assert "bad sample should be skipped" not in text


def test_pretrain_parquet_directory_sharding_is_row_level(monkeypatch, tmp_path) -> None:
    class FakeColumn:
        def __init__(self, rows):
            self._rows = rows

        def to_pylist(self):
            return self._rows

    class FakeBatch:
        def __init__(self, rows):
            self._rows = rows

        def column(self, index):
            assert index == 0
            return FakeColumn(self._rows)

    rows_by_file = {
        "part-0.parquet": [[0], [1], [2]],
        "part-1.parquet": [[3], [4], [5]],
    }

    class FakeParquetFile:
        def __init__(self, path):
            self._name = Path(path).name

        def iter_batches(self, columns, batch_size):
            assert columns == ["input_ids"]
            assert batch_size == 2048
            yield FakeBatch(rows_by_file[self._name])

    monkeypatch.setattr(pretrain_module.pq, "ParquetFile", FakeParquetFile)
    parquet_files = [tmp_path / "part-0.parquet", tmp_path / "part-1.parquet"]
    ds = PreTrainDataset.__new__(PreTrainDataset)

    rows = []
    shard_lengths = []
    for shard_id in range(4):
        shard_rows = list(ds._iter_token_ids_parquet_shards(shard_id, 4, parquet_files))
        shard_lengths.append(len(shard_rows))
        rows.extend(shard_rows)

    assert sorted(row[0] for row in rows) == list(range(6))
    assert all(length > 0 for length in shard_lengths)
