from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


def _normalize_pack_bin_schedule(
    schedule: list[Any] | None, fallback_pack_bin_size: int
) -> list[tuple[int | None, int]]:
    """解析为 [(until_exclusive, size), ..., (None, final_size)]；until 为累计打包样本数上界（不含）。"""
    if not schedule:
        return [(None, int(fallback_pack_bin_size))]
    out: list[tuple[int | None, int]] = []
    last_until = 0
    n = len(schedule)
    for i, item in enumerate(schedule):
        if not isinstance(item, dict):
            raise TypeError(f"pack_bin_schedule[{i}] 应为 JSON 对象，收到 {type(item).__name__}")
        sz = item.get("pack_bin_size")
        if sz is None:
            raise ValueError(f"pack_bin_schedule[{i}] 缺少 pack_bin_size")
        sz = int(sz)
        if sz <= 0:
            raise ValueError(f"pack_bin_size 须为正整数，收到 {sz}")
        if i < n - 1:
            u = item.get("until_index")
            if u is None:
                raise ValueError(f"pack_bin_schedule[{i}] 非最后阶段必须包含 until_index（累计样本上界）")
            u = int(u)
            if u <= last_until:
                raise ValueError(
                    f"until_index 须严格递增：第 {i} 项为 {u}，上一阶段上界为 {last_until}"
                )
            out.append((u, sz))
            last_until = u
        else:
            if item.get("until_index") is not None:
                raise ValueError(
                    "pack_bin_schedule 最后一项不应包含 until_index（表示该阶段一直装到语料耗尽）"
                )
            out.append((None, sz))
    return out


def _iter_jsonl_objects(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL，避免 ``datasets.load_dataset`` 在本地生成 Arrow 缓存占满磁盘。"""
    p = Path(path)
    with p.open(encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise TypeError(
                    f"{p}:{lineno}: 每行须为 JSON 对象，收到 {type(obj).__name__}"
                )
            yield obj


class PreTrainDataset(IterableDataset):
    """预训练流式数据集：支持 JSONL 在线分词，或直接读取预 tokenized Arrow 数据。"""
    _arrow_dataset_cache: dict[str, Any] = {}

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        pack_bin_size: int,
        pack_bin_schedule: list[Any] | None = None,
    ) -> None:
        self.data_path = str(data_path)
        self.tokenizer = tokenizer
        self._stages = _normalize_pack_bin_schedule(pack_bin_schedule, pack_bin_size)
        self.pack_bin_size = max(s[1] for s in self._stages)
        self._sep_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
        self._use_fast = bool(getattr(tokenizer, "is_fast", False))
        self._fast_tokenizer = None
        if self._use_fast:
            # 优先直接走底层 tokenizers 接口，减少 Python 层封装开销。
            self._fast_tokenizer = getattr(tokenizer, "_tokenizer", None)

    def _iter_token_ids_jsonl(
        self, shard_id: int, num_shards: int
    ) -> Iterator[list[int]]:
        for i, row in enumerate(_iter_jsonl_objects(self.data_path)):
            if (i % num_shards) != shard_id:
                continue
            text = row.get("text", "")
            if not text or not str(text).strip():
                continue
            text = str(text)
            if self._use_fast and self._fast_tokenizer is not None:
                ids = self._fast_tokenizer.encode(text, add_special_tokens=False).ids
            else:
                enc = self.tokenizer.encode(text, add_special_tokens=False)
                ids = enc if isinstance(enc, list) else list(enc)
            if ids:
                yield ids

    def _iter_token_ids_arrow(
        self, shard_id: int, num_shards: int
    ) -> Iterator[list[int]]:
        ds = PreTrainDataset._arrow_dataset_cache.get(self.data_path)
        if ds is None:
            ds = load_from_disk(self.data_path)
            PreTrainDataset._arrow_dataset_cache[self.data_path] = ds
        if "input_ids" not in ds.column_names:
            raise KeyError(
                f"{self.data_path} 缺少 input_ids 列，请先执行预 tokenization。"
            )
        shard = ds.shard(num_shards=num_shards, index=shard_id, contiguous=False)
        for row in shard:
            ids = row.get("input_ids")
            if not ids:
                continue
            yield list(ids)

    @staticmethod
    def _shard_info() -> tuple[int, int]:
        """返回 (shard_id, num_shards)，同时切分 DDP rank 与 dataloader workers。"""
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        return shard_id, max(1, num_shards)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shard_id, num_shards = self._shard_info()

        stage_idx = 0
        emitted = 0
        buffer: list[int] = []
        seen_docs = 0
        skipped_empty = 0
        first_doc = True

        def maybe_advance_stage() -> None:
            nonlocal stage_idx
            while stage_idx < len(self._stages) - 1:
                until_excl, _sz = self._stages[stage_idx]
                if until_excl is None or emitted < until_excl:
                    break
                stage_idx += 1

        def current_chunk_size() -> int:
            _until_excl, sz = self._stages[stage_idx]
            return int(sz)

        p = Path(self.data_path)
        use_arrow = p.is_dir() and (p / "dataset_info.json").exists()
        ids_iter = (
            self._iter_token_ids_arrow(shard_id, num_shards)
            if use_arrow
            else self._iter_token_ids_jsonl(shard_id, num_shards)
        )

        ids_buffer: deque[list[int]] = deque(maxlen=10)

        def buffered_ids_iter() -> Iterator[list[int]]:
            for ids in ids_iter:
                ids_buffer.append(ids)
                if len(ids_buffer) >= 5:
                    yield ids_buffer.popleft()
            while ids_buffer:
                yield ids_buffer.popleft()

        for ids in buffered_ids_iter():

            seen_docs += 1
            if not first_doc:
                buffer.append(self._sep_id)
            buffer.extend(ids)
            first_doc = False

            maybe_advance_stage()
            while len(buffer) >= current_chunk_size():
                csz = current_chunk_size()
                piece = buffer[:csz]
                del buffer[:csz]
                input_ids = torch.tensor(piece, dtype=torch.long)
                labels = input_ids.clone()
                yield {"input_ids": input_ids, "labels": labels}
                emitted += 1
                maybe_advance_stage()

        # 语料结束后，尾段不足一个 chunk 仍产出一条短样本。
        if buffer:
            input_ids = torch.tensor(buffer, dtype=torch.long)
            labels = input_ids.clone()
            yield {"input_ids": input_ids, "labels": labels}
            emitted += 1

        logger.info(
            "PreTrainDataset(streaming): source=%s docs=%s emitted=%s max_chunk=%s skipped_empty=%s shard=%s/%s stages=%s",
            "arrow" if use_arrow else "jsonl",
            seen_docs,
            emitted,
            self.pack_bin_size,
            skipped_empty,
            shard_id,
            num_shards,
            self._stages,
        )
