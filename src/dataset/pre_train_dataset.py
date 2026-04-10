from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

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


class PreTrainDataset(Dataset):
    """预训练：Token 流式 packing（标准长序列预训练）。

    按 **JSONL 文件中文档出现顺序**（与 preprocess 写出顺序一致）依次读取，将所有文档首尾相接成
    **一条 Token 流**，相邻文档之间只插入 **一个** ``<|endoftext|>`` 分隔符，
    再按桶容量做**硬截断**：从流头起每 ``pack_bin_size``（或 ``pack_bin_schedule`` 中各阶段长度）个
    token 切成一条样本；句子/文档可任意跨越桶边界。最后一段允许短于阶段块长，
    后续由 data collator 动态 padding。

    ``pack_bin_schedule``：表示「已产出样本数达到 until_index 前用该阶段块长」，流上读指针连续推进，
    不在阶段边界对齐到文档边界。
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        pack_bin_size: int,
        pack_bin_schedule: list[Any] | None = None,
    ):
        self.data_path = str(data_path)
        self.tokenizer = tokenizer
        self._stages = _normalize_pack_bin_schedule(pack_bin_schedule, pack_bin_size)
        self.pack_bin_size = max(s[1] for s in self._stages)
        sep_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        doc_token_ids: list[list[int]] = []
        skipped_empty = 0
        for row in _iter_jsonl_objects(self.data_path):
            text = row.get("text", "")
            if not text or not str(text).strip():
                skipped_empty += 1
                continue
            enc = tokenizer(
                str(text),
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            ids = enc["input_ids"]
            if not ids:
                skipped_empty += 1
                continue
            doc_token_ids.append(ids)

        token_stream = PreTrainDataset._build_token_stream(doc_token_ids, int(sep_id))
        packed, stream_exhausted_early = PreTrainDataset._stream_pack_staged(token_stream, self._stages)
        if stream_exhausted_early:
            logger.warning(
                "PreTrainDataset：Token 流在达到 pack_bin_schedule 全部阶段前已耗尽，"
                "后续阶段未再产出样本（可检查语料量或 schedule）"
            )

        self._packed: list[dict[str, torch.Tensor]] = []
        for b in packed:
            input_ids = torch.tensor(b, dtype=torch.long)
            labels = input_ids.clone()
            self._packed.append({"input_ids": input_ids, "labels": labels})

        stream_tokens = len(token_stream)
        if pack_bin_schedule:
            logger.info(
                "PreTrainDataset stream packing (staged): raw_docs=%s stream_tokens=%s samples=%s "
                "max_chunk=%s skipped_empty=%s stages=%s",
                len(doc_token_ids),
                stream_tokens,
                len(self._packed),
                self.pack_bin_size,
                skipped_empty,
                self._stages,
            )
        else:
            logger.info(
                "PreTrainDataset stream packing: raw_docs=%s stream_tokens=%s samples=%s "
                "pack_bin_size=%s skipped_empty=%s",
                len(doc_token_ids),
                stream_tokens,
                len(self._packed),
                self.pack_bin_size,
                skipped_empty,
            )

    def __len__(self) -> int:
        return len(self._packed)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._packed[index]

    @staticmethod
    def _build_token_stream(doc_token_ids: list[list[int]], sep_token_id: int) -> list[int]:
        """所有文档 token 首尾相接，相邻文档之间只插一个分隔符（不占文档前导）。"""
        stream: list[int] = []
        first = True
        for ids in doc_token_ids:
            if not ids:
                continue
            if not first:
                stream.append(sep_token_id)
            stream.extend(ids)
            first = False
        return stream

    @staticmethod
    def _stream_pack_staged(
        stream: list[int],
        stages: list[tuple[int | None, int]],
    ) -> tuple[list[list[int]], bool]:
        """
        从流头连续切分；第二项为 True 表示在某 until 阶段尚未凑满约定样本数时流已耗尽。
        注意：此处不做样本内右侧补齐，样本可变长；padding 由 collator 统一动态处理。
        """
        n = len(stream)
        pos = 0
        chunks: list[list[int]] = []
        emitted = 0

        for until_excl, chunk_size in stages:
            if chunk_size <= 0:
                raise ValueError(f"stream pack chunk_size 须为正，收到 {chunk_size}")

            if until_excl is not None:
                need = until_excl - emitted
                if need <= 0:
                    continue
                for _ in range(need):
                    if pos >= n:
                        return chunks, True
                    end = pos + chunk_size
                    piece = stream[pos:end]
                    pos = end
                    chunks.append(list(piece))
                    emitted += 1
                continue

            while pos < n:
                end = pos + chunk_size
                piece = stream[pos:end]
                pos = end
                chunks.append(list(piece))
                emitted += 1
            return chunks, False

        return chunks, False


class StreamingPreTrainDataset(IterableDataset):
    """预训练流式数据集：逐行读取 JSONL 并在线 packing，避免全量读入内存后再开训。"""

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

        for i, row in enumerate(_iter_jsonl_objects(self.data_path)):
            if (i % num_shards) != shard_id:
                continue

            text = row.get("text", "")
            if not text or not str(text).strip():
                skipped_empty += 1
                continue

            enc = self.tokenizer(
                str(text),
                add_special_tokens=False,
                truncation=False,
                padding=False,
                verbose=False,
            )
            ids = enc["input_ids"]
            if not ids:
                skipped_empty += 1
                continue

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
            "StreamingPreTrainDataset: docs=%s emitted=%s max_chunk=%s skipped_empty=%s shard=%s/%s stages=%s",
            seen_docs,
            emitted,
            self.pack_bin_size,
            skipped_empty,
            shard_id,
            num_shards,
            self._stages,
        )
