import json

import pyarrow.parquet as pq

from scripts import to_arrow


class _FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[len(text)] for text in texts]}


def _write_jsonl(path, texts):
    path.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts),
        encoding="utf-8",
    )


def test_rebuild_removes_stale_parquet_shards(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "tokenized"
    monkeypatch.setattr(
        to_arrow,
        "get_auto_tokenizer_local",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )

    _write_jsonl(source, ["one", "two", "three", "four", "five"])
    to_arrow.build_arrow_dataset(
        str(source),
        str(output),
        "unused",
        batch_size=2,
        rows_per_file=2,
    )
    assert sorted(path.name for path in output.glob("part-*.parquet")) == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]

    _write_jsonl(source, ["replacement"])
    to_arrow.build_arrow_dataset(
        str(source),
        str(output),
        "unused",
        batch_size=2,
        rows_per_file=2,
    )

    shards = sorted(output.glob("part-*.parquet"))
    assert [path.name for path in shards] == ["part-00000.parquet"]
    assert pq.read_table(shards[0]).column("input_ids").to_pylist() == [[11]]
    info = json.loads((output / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["splits"]["train"]["num_examples"] == 1
