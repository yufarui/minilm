from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.ref_model import get_auto_tokenizer_local

"""
python -m scripts.to_arrow \
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
    """JSONL -> HF Arrow Dataset（含 input_ids），供预训练阶段直接读取。"""
    print(f"Loading JSONL from {jsonl_path}")
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    total_before = len(ds)
    print(f"Total samples: {total_before}")

    tokenizer = get_auto_tokenizer_local(tokenizer_name_or_path, trust_remote_code=True)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        ids = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
        return {"input_ids": ids}

    ds = ds.filter(
        lambda x: bool(str(x.get("text", "")).strip()),
        num_proc=num_proc,
        desc="Filtering empty texts",
    )
    total_after_filter = len(ds)
    print(f"After filtering: {total_after_filter} samples ({total_before - total_after_filter} removed)")

    ds = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=[c for c in ds.column_names if c != "text"],
        desc="Tokenizing",
    )
    ds = ds.remove_columns(["text"]) if "text" in ds.column_names else ds

    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample input_ids length: {len(sample['input_ids'])}")
        n_est = min(100, len(ds))
        est_tokens = 0
        for row in tqdm(ds.select(range(n_est)), total=n_est, desc="Estimating tokens (first 100)"):
            est_tokens += len(row["input_ids"])
        print(f"Estimated total tokens: {est_tokens} (first {n_est} samples)")
    else:
        print("Warning: dataset is empty after filtering/tokenization.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"Saved tokenized dataset to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL pretrain corpus to tokenized Arrow dataset.")
    parser.add_argument("--jsonl_path", required=True, help="Input JSONL path, each row contains text field.")
    parser.add_argument("--output_path", required=True, help="Output directory for save_to_disk().")
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
