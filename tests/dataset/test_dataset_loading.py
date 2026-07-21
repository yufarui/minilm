import json
import pytest
from itertools import islice

import pyarrow as pa
import pyarrow.parquet as pq

import src.dataset.pre_train_dataset as pre_train_dataset
from src.dataset.dpo_dataset import DPODataset
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


def test_pretrain_parquet_shards_are_streamed(tmp_path, monkeypatch) -> None:
    shard_path = tmp_path / "part-00000.parquet"
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array(
                    [[1, 2, 3], [4, 5], []],
                    type=pa.list_(pa.int32()),
                )
            }
        ),
        shard_path,
    )

    real_parquet_file = pq.ParquetFile
    iter_batch_calls = 0

    class StreamingOnlyParquetFile:
        def __init__(self, path) -> None:
            self._inner = real_parquet_file(path)

        def read(self, *args, **kwargs):
            raise AssertionError("parquet shards must not be materialized with read()")

        def iter_batches(self, *args, **kwargs):
            nonlocal iter_batch_calls
            iter_batch_calls += 1
            yield from self._inner.iter_batches(*args, **kwargs)

    monkeypatch.setattr(pre_train_dataset.pq, "ParquetFile", StreamingOnlyParquetFile)

    ds = object.__new__(PreTrainDataset)
    rows = list(ds._iter_token_ids_parquet_shards(0, 1, [shard_path]))

    assert rows == [[1, 2, 3], [4, 5]]
    assert iter_batch_calls == 1


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
