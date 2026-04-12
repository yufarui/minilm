from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from torch.utils.data import IterableDataset

from src.dataset.pre_train_dataset import PreTrainDataset, _iter_jsonl_objects

logger = logging.getLogger(__name__)

# 训练入口不传参时使用以下默认；需调整 packing 策略时在构造 SFTDataset 时显式传入。
DEFAULT_PACK_SORT_ORDER = "file_order"
DEFAULT_MAX_OPEN_BINS = 512
DEFAULT_SORT_BUFFER_SIZE = 4096


class SFTDataset(IterableDataset):
    """对话 SFT：JSONL 流式逐行读取，边编码边打包，不一次性载入全量再 packing。

    - ``pack_sort_order=file_order``：与预训练类似，按行顺序在线 first-fit，内存由 ``max_open_bins`` 约束。
    - ``shortest_first`` / ``longest_first``：按块排序（块大小 ``sort_buffer_size``），避免整表排序占满内存。
    """

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minilm，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minilm，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minilm, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minilm, a small but useful language model.",
    ]

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        pack_bin_size: int,
        pack_sort_order: str = DEFAULT_PACK_SORT_ORDER,
        max_open_bins: int = DEFAULT_MAX_OPEN_BINS,
        sort_buffer_size: int = DEFAULT_SORT_BUFFER_SIZE,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pack_bin_size = int(pack_bin_size)
        self.pack_sort_order = pack_sort_order
        self.max_open_bins = max(1, int(max_open_bins))
        self.sort_buffer_size = max(1, int(sort_buffer_size))

        sep_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self._sep_id = int(sep_id)
        self.jsonl_path = str(jsonl_path)

        self.bos_enc_id = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.eos_enc_id = tokenizer.encode(f"{tokenizer.eos_token}\n", add_special_tokens=False)

        self.add_system_ratio = 0.2

        logger.info(
            "SFTDataset(streaming): path=%s pack_bin_size=%s sort=%s max_open_bins=%s sort_buffer_size=%s",
            self.jsonl_path,
            self.pack_bin_size,
            self.pack_sort_order,
            self.max_open_bins,
            self.sort_buffer_size,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shard_id, num_shards = PreTrainDataset._shard_info()
        if self.pack_sort_order in ("shortest_first", "longest_first"):
            yield from self._iter_sort_buffer_pack(shard_id, num_shards)
        else:
            if self.pack_sort_order != "file_order":
                logger.warning(
                    "未知 pack_sort_order=%r，按 file_order 流式处理",
                    self.pack_sort_order,
                )
            yield from self._iter_streaming_first_fit(shard_id, num_shards)

    def _iter_jsonl_encoded(
        self, shard_id: int, num_shards: int
    ) -> Iterator[tuple[list[int], list[int]]]:
        for i, row in enumerate(_iter_jsonl_objects(self.jsonl_path)):
            if (i % num_shards) != shard_id:
                continue
            conversations = row.get("conversations", [])
            if not conversations:
                continue
            enc = self._encode_conversation(conversations)
            if enc is None:
                continue
            input_ids, labels = enc
            if not input_ids:
                continue
            if len(input_ids) > self.pack_bin_size:
                input_ids = input_ids[: self.pack_bin_size]
                labels = labels[: self.pack_bin_size]
            yield input_ids, labels

    def _iter_streaming_first_fit(
        self, shard_id: int, num_shards: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        bins_in: list[list[int]] = []
        bins_lab: list[list[int]] = []
        emitted = 0
        seen_docs = 0

        def try_place(doc_in: list[int], doc_lab: list[int]) -> bool:
            for b_in, b_lab in zip(bins_in, bins_lab):
                gap = 1 if b_in else 0
                if len(b_in) + gap + len(doc_in) <= self.pack_bin_size:
                    if gap:
                        b_in.append(self._sep_id)
                        b_lab.append(-100)
                    b_in.extend(doc_in)
                    b_lab.extend(doc_lab)
                    return True
            return False

        def emit_bin(idx: int) -> dict[str, torch.Tensor]:
            nonlocal emitted
            b_in = bins_in.pop(idx)
            b_lab = bins_lab.pop(idx)
            emitted += 1
            return {
                "input_ids": torch.tensor(b_in, dtype=torch.long),
                "labels": torch.tensor(b_lab, dtype=torch.long),
            }

        for input_ids, labels in self._iter_jsonl_encoded(shard_id, num_shards):
            seen_docs += 1
            while True:
                if try_place(input_ids, labels):
                    break
                if len(bins_in) >= self.max_open_bins:
                    yield emit_bin(0)
                    continue
                bins_in.append(list(input_ids))
                bins_lab.append(list(labels))
                break

        while bins_in:
            yield emit_bin(0)

        logger.info(
            "SFTDataset(streaming first-fit): docs=%s emitted=%s shard=%s/%s",
            seen_docs,
            emitted,
            shard_id,
            num_shards,
        )

    def _iter_sort_buffer_pack(
        self, shard_id: int, num_shards: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        buf: list[tuple[list[int], list[int]]] = []
        emitted = 0
        rev = self.pack_sort_order == "longest_first"

        def flush() -> Iterator[dict[str, torch.Tensor]]:
            nonlocal emitted, buf
            if not buf:
                return
            buf.sort(key=lambda p: len(p[0]), reverse=rev)
            packed = SFTDataset._pack_sft_pairs_into_bins(buf, self.pack_bin_size, self._sep_id)
            buf.clear()
            for input_ids, labels in packed:
                emitted += 1
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }

        for pair in self._iter_jsonl_encoded(shard_id, num_shards):
            buf.append(pair)
            if len(buf) >= self.sort_buffer_size:
                yield from flush()
        yield from flush()

        logger.info(
            "SFTDataset(sort-buffer pack): emitted=%s order=%s shard=%s/%s",
            emitted,
            self.pack_sort_order,
            shard_id,
            num_shards,
        )

    @staticmethod
    def _tool_calls_fill(conv: List[Dict[str, Any]]):
        """
        chat_template.jinja 需要 assistant.tool_calls 为列表；
        JSONL 里常为 JSON 字符串，否则 Jinja 会按字符迭代。
        若存在非空但非法的 tool_calls JSON，返回 False（应跳过该条样本）。
        """
        for msg in conv:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if "tool_calls" not in msg:
                continue
            raw = msg["tool_calls"]
            if isinstance(raw, list):
                continue
            if not isinstance(raw, str):
                logger.warning("assistant.tool_calls 类型无效（%s），跳过该字段", type(raw).__name__)
                msg.pop("tool_calls", None)
                continue
            s = raw.strip()
            try:
                msg["tool_calls"] = json.loads(s)
            except json.JSONDecodeError as e:
                logger.warning("assistant.tool_calls JSON 无效，跳过该条: %s", e)
                msg.pop("tool_calls", None)

    def _encode_conversation(self, conversations: List[Dict[str, Any]]) -> tuple[list[int], list[int]] | None:
        """返回 (input_ids, labels)；逻辑与原先 __getitem__ 一致，不修改原始样本。"""
        conv = copy.deepcopy(conversations)
        if not conv:
            return None

        first_message = conv[0]
        if first_message.get("role") == "system" and not first_message.get("content"):
            first_message["content"] = random.choice(self.SYSTEM_PROMPTS)

        tools = None
        if first_message.get("role") == "system":
            raw_tools = first_message.get("tools")
            if raw_tools is not None:
                if isinstance(raw_tools, list):
                    tools = raw_tools
                elif isinstance(raw_tools, str):
                    s = raw_tools.strip()
                    if s:
                        try:
                            tools = json.loads(s)
                        except json.JSONDecodeError as e:
                            logger.warning("system.tools JSON 无效，跳过该条: %s", e)
                            return None

        if first_message.get("role") != "system":
            if random.random() < self.add_system_ratio:
                conv.insert(0, {"role": "system", "content": random.choice(self.SYSTEM_PROMPTS)})

        self._tool_calls_fill(conv)

        raw = self.tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
            open_think=False,
        )

        prompt = raw.detach().cpu().flatten().tolist()

        labels = self.generate_labels(prompt)
        return prompt, labels

    def generate_labels(self, prompt: list[int]) -> list[int]:
        labels = [-100] * len(prompt)
        i = 0
        bos, eos = self.bos_enc_id, self.eos_enc_id
        lbos, leos = len(bos), len(eos)
        while i < len(prompt):
            if prompt[i : i + lbos] == bos:
                start = i + lbos
                j = start
                while j < len(prompt):
                    labels[j] = prompt[j]
                    if prompt[j : j + leos] == eos:
                        break
                    j += 1
                i = j + leos if j < len(prompt) else len(prompt)
            else:
                i += 1
        return labels

    @staticmethod
    def _pack_sft_pairs_into_bins(
        doc_pairs: list[tuple[list[int], list[int]]],
        pack_bin_size: int,
        sep_token_id: int,
    ) -> list[tuple[list[int], list[int]]]:
        """First-fit：依次尝试放入已有 bin；包内用 sep_token 拼接 input，labels 在分隔处为 -100。"""
        bins_in: list[list[int]] = []
        bins_lab: list[list[int]] = []
        for input_ids, labels in doc_pairs:
            if not input_ids:
                continue
            if len(input_ids) > pack_bin_size:
                input_ids = input_ids[:pack_bin_size]
                labels = labels[:pack_bin_size]
            placed = False
            for b_in, b_lab in zip(bins_in, bins_lab):
                gap = 1 if b_in else 0
                if len(b_in) + gap + len(input_ids) <= pack_bin_size:
                    if gap:
                        b_in.append(sep_token_id)
                        b_lab.append(-100)
                    b_in.extend(input_ids)
                    b_lab.extend(labels)
                    placed = True
                    break
            if not placed:
                bins_in.append(list(input_ids))
                bins_lab.append(list(labels))
        return list(zip(bins_in, bins_lab))
