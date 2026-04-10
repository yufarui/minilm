from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.ref_model import get_auto_tokenizer_local

"""
python -m ./scripts/to_arrow \
  --jsonl_path data/pretrain/pretrain_train.jsonl \
  --output_path data/pretrain/pretrain_train_arrow \
  --tokenizer_name_or_path tokenizer/minilm \
  --num_proc 16 \
  --batch_size 1024
"""


def build_arrow_dataset(
    jsonl_path: str,
    output_path: str,
    tokenizer_name_or_path: str,
    num_proc: int = 16,
    batch_size: int = 1024,
) -> None:
    """JSONL -> Parquet(input_ids)；流式处理，避免 datasets.load_dataset 的中间缓存占盘。"""
    del num_proc  # 预留参数，当前实现为单进程流式写盘。
    print(f"Streaming JSONL from {jsonl_path}")
    tokenizer = get_auto_tokenizer_local(tokenizer_name_or_path, trust_remote_code=True)

    out = Path(output_path)
    out_file = out if out.suffix == ".parquet" else (out / "tokenized.parquet")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    writer: pq.ParquetWriter | None = None

    total_before = 0
    total_after = 0
    dropped = 0
    first_len: int | None = None
    first_100_token_sum = 0
    first_100_seen = 0
    tokenized_rows: list[list[int]] = []
    write_batch_size = max(2048, batch_size)

    def flush_rows() -> None:
        nonlocal writer, tokenized_rows
        if not tokenized_rows:
            return
        arr = pa.array(tokenized_rows, type=pa.list_(pa.int32()))
        table = pa.Table.from_arrays([arr], names=["input_ids"], schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(str(out_file), schema=schema, compression="zstd")
        writer.write_table(table)
        tokenized_rows = []

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
            if len(tokenized_rows) >= write_batch_size:
                flush_rows()

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
            flush_rows()
        else:
            flush_rows()

    if writer is not None:
        writer.close()

    print(f"Total lines: {total_before}")
    print(f"After filtering/tokenizing: {total_after} samples ({dropped} removed)")
    if first_len is not None:
        print(f"Sample input_ids length: {first_len}")
        print(f"Estimated total tokens: {first_100_token_sum} (first {first_100_seen} samples)")
    else:
        print("Warning: dataset is empty after filtering/tokenization.")
    print(f"Saved tokenized parquet to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL pretrain corpus to tokenized parquet dataset.")
    parser.add_argument("--jsonl_path", required=True, help="Input JSONL path, each row contains text field.")
    parser.add_argument("--output_path", required=True, help="Output parquet file path or output directory.")
    parser.add_argument("--tokenizer_name_or_path", default="tokenizer/minilm", help="Tokenizer path or name.")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes for map/filter.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for tokenization map.")
    args = parser.parse_args()

    build_arrow_dataset(
        jsonl_path=args.jsonl_path,
        output_path=args.output_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
