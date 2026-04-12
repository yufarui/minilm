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

# 流式 first-fit 时最多保留的未完成 bin；超出则 FIFO 产出，避免内存无限增长。
DEFAULT_MAX_OPEN_BINS = 512


class SFTDataset(IterableDataset):
    """对话 SFT：JSONL 流式逐行读取，多段对话 **first-fit packing** 拼成一条样本。

    - 单条对话 **不拆到两个 pack**；过长则 **截断至 pack_bin_size** 以单独占满一包（与显存上限一致）。
    - 包与包之间插入 ``<|endoftext|>``，对应 ``labels=-100``，供 ``TrainDataCollator`` 在分隔处重置
      ``position_ids``、构造 packing 注意力掩码（与预训练 packing 语义一致）。
    - 无 bin_scheduler；仅按 JSONL 顺序流式 first-fit，内存由 ``max_open_bins`` 约束。
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
        max_open_bins: int = DEFAULT_MAX_OPEN_BINS,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pack_bin_size = int(pack_bin_size)
        self.max_open_bins = max(1, int(max_open_bins))
        self.jsonl_path = str(jsonl_path)

        self._sep_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))

        self.bos_enc_id = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.eos_enc_id = tokenizer.encode(f"{tokenizer.eos_token}\n", add_special_tokens=False)

        self.add_system_ratio = 0.2

        logger.info(
            "SFTDataset(packing): path=%s pack_bin_size=%s max_open_bins=%s",
            self.jsonl_path,
            self.pack_bin_size,
            self.max_open_bins,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shard_id, num_shards = PreTrainDataset._shard_info()
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
            "SFTDataset(packing): shard=%s/%s docs=%s emitted=%s",
            shard_id,
            num_shards,
            seen_docs,
            emitted,
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
        """返回 (input_ids, labels)；不修改原始样本。"""
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
        prompt = self._coerce_chat_template_output_to_ids(raw)

        labels = self.generate_labels(prompt)
        return prompt, labels

    def _coerce_chat_template_output_to_ids(self, raw: Any) -> list[int]:
        """``apply_chat_template(tokenize=True)`` 在不同版本下可能返回 list、Tensor、或 BatchEncoding。"""
        if isinstance(raw, str):
            return self.tokenizer.encode(raw, add_special_tokens=False)
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().flatten().tolist()
        if isinstance(raw, dict):
            inner = raw.get("input_ids")
            if inner is not None:
                return self._coerce_chat_template_output_to_ids(inner)
        if hasattr(raw, "input_ids"):
            inner = raw["input_ids"]
            return self._coerce_chat_template_output_to_ids(inner)
        if isinstance(raw, (list, tuple)):
            return [int(x) for x in raw]
        try:
            import numpy as np

            if isinstance(raw, np.ndarray):
                return raw.astype(np.int64).flatten().tolist()
        except ImportError:
            pass
        raise TypeError(
            f"无法将 apply_chat_template 输出转为 token id 列表，类型={type(raw).__name__}"
        )

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
