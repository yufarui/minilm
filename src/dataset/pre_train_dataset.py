from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

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
        self.samples = load_dataset("json", data_files=self.data_path, split="train")

        doc_token_ids: list[list[int]] = []
        skipped_empty = 0
        for i in range(len(self.samples)):
            text = self.samples[i].get("text", "")
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
