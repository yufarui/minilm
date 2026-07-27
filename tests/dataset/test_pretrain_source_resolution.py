"""Regression: parquet shards must win over leftover HF save_to_disk markers."""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset.pretrain_source import (
    remove_legacy_hf_disk_artifacts,
    resolve_pretrain_source,
)


def test_resolve_prefers_parquet_shards_over_stale_hf_state(tmp_path: Path) -> None:
    """Rebuild path: part-*.parquet + dataset_info.json + leftover state.json."""
    (tmp_path / "dataset_info.json").write_text(
        json.dumps({"builder_name": "streaming_parquet_shards"}),
        encoding="utf-8",
    )
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data-00000-of-00001.arrow").write_bytes(b"stale")
    part = tmp_path / "part-00000.parquet"
    part.write_bytes(b"parquet-bytes")

    source, files = resolve_pretrain_source(tmp_path)
    assert source == "parquet-shards"
    assert files == [part]


def test_resolve_uses_arrow_when_no_parquet_shards(tmp_path: Path) -> None:
    (tmp_path / "dataset_info.json").write_text("{}", encoding="utf-8")
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data-00000-of-00001.arrow").write_bytes(b"ok")

    source, files = resolve_pretrain_source(tmp_path)
    assert source == "arrow"
    assert files == []


def test_resolve_jsonl_file(tmp_path: Path) -> None:
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"text":"hi"}\n', encoding="utf-8")
    source, files = resolve_pretrain_source(jsonl)
    assert source == "jsonl"
    assert files == []


def test_remove_legacy_hf_disk_artifacts(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "dataset_dict.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data-00000-of-00001.arrow").write_bytes(b"stale")
    nested = tmp_path / "nested_split"
    nested.mkdir()
    (nested / "shard.arrow").write_bytes(b"stale")
    (tmp_path / "part-00000.parquet").write_bytes(b"keep")

    removed = remove_legacy_hf_disk_artifacts(tmp_path)

    assert (tmp_path / "state.json").as_posix() in {Path(p).as_posix() for p in removed} or any(
        p.endswith("state.json") for p in removed
    )
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "dataset_dict.json").exists()
    assert not (tmp_path / "data-00000-of-00001.arrow").exists()
    assert not nested.exists()
    assert (tmp_path / "part-00000.parquet").exists()
