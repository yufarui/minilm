from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.ref_model import get_auto_tokenizer_local

"""
uv run scripts/to_arrow.py \
  --jsonl_path data/pretrain/pretrain_train.jsonl \
  --output_path data/pretrain/pretrain_train_arrow \
  --tokenizer_name_or_path tokenizer/minilm \
  --batch_size 1024 \
  --rows_per_file 500000
"""


def build_arrow_dataset(
    jsonl_path: str,
    output_path: str,
    tokenizer_name_or_path: str,
    batch_size: int = 1024,
    rows_per_file: int = 500_000,
) -> None:
    """流式处理 JSONL，输出多个 Parquet 分片，避免单文件 I/O 竞争。"""
    print(f"Streaming JSONL from {jsonl_path}")
    tokenizer = get_auto_tokenizer_local(tokenizer_name_or_path, trust_remote_code=True)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])

    tokenized_rows: list[list[int]] = []
    total_rows_written = 0
    file_index = 0
    total_before = 0
    total_after = 0
    dropped = 0
    first_len: int | None = None
    first_100_token_sum = 0
    first_100_seen = 0

    def write_shard(rows: list[list[int]], idx: int) -> None:
        nonlocal total_rows_written
        if not rows:
            return
        out_file = out_dir / f"part-{idx:05d}.parquet"
        arr = pa.array(rows, type=pa.list_(pa.int32()))
        table = pa.Table.from_arrays([arr], schema=schema)
        pq.write_table(table, out_file, compression="zstd")
        total_rows_written += len(rows)
        print(f"Written {len(rows)} rows to {out_file}")

    with open(jsonl_path, encoding="utf-8") as fp:
        text_batch: list[str] = []
        for line in tqdm(fp, desc="Reading+tokenizing", unit="lines"):
            total_before += 1
            line = line.strip()
            if not line:
                dropped += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            text = str(obj.get("text", "")).strip()
            if not text:
                dropped += 1
                continue

            text_batch.append(text)
            if len(text_batch) < batch_size:
                continue

            ids_batch = tokenizer(
                text_batch,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            for ids in ids_batch:
                if not ids:
                    dropped += 1
                    continue
                row = [int(x) for x in ids]
                tokenized_rows.append(row)
                total_after += 1
                if first_len is None:
                    first_len = len(row)
                if first_100_seen < 100:
                    first_100_token_sum += len(row)
                    first_100_seen += 1
            text_batch = []
            if len(tokenized_rows) >= rows_per_file:
                write_shard(tokenized_rows, file_index)
                tokenized_rows = []
                file_index += 1

        if text_batch:
            ids_batch = tokenizer(
                text_batch,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            for ids in ids_batch:
                if not ids:
                    dropped += 1
                    continue
                row = [int(x) for x in ids]
                tokenized_rows.append(row)
                total_after += 1
                if first_len is None:
                    first_len = len(row)
                if first_100_seen < 100:
                    first_100_token_sum += len(row)
                    first_100_seen += 1

        if tokenized_rows:
            write_shard(tokenized_rows, file_index)
            tokenized_rows = []
            file_index += 1

    info = {
        "builder_name": "streaming_parquet_shards",
        "config_name": "default",
        "version": {"version_str": "1.0.0"},
        "splits": {"train": {"name": "train", "num_examples": total_after}},
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    print(f"Total lines: {total_before}")
    print(f"After filtering: {total_after} samples ({dropped} removed)")
    if first_len is not None:
        print(f"Sample input_ids length: {first_len}")
        print(f"Estimated total tokens: {first_100_token_sum} (first {first_100_seen} samples)")
    else:
        print("Warning: dataset is empty after filtering/tokenization.")
    print(f"Written {total_rows_written} rows into {file_index} shards in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL pretrain corpus to tokenized parquet dataset.")
    parser.add_argument("--jsonl_path", required=True, help="Input JSONL path, each row contains text field.")
    parser.add_argument("--output_path", required=True, help="Output directory for parquet shards.")
    parser.add_argument("--tokenizer_name_or_path", default="tokenizer/minilm", help="Tokenizer path or name.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for tokenization map.")
    parser.add_argument("--rows_per_file", type=int, default=500000, help="Max rows per parquet shard.")
    args = parser.parse_args()

    build_arrow_dataset(
        jsonl_path=args.jsonl_path,
        output_path=args.output_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        batch_size=args.batch_size,
        rows_per_file=args.rows_per_file,
    )


if __name__ == "__main__":
    main()
