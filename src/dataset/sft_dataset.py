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


class SFTDataset(IterableDataset):
    """对话 SFT：JSONL **一行一条样本**，流式读取，**不做 packing、不做截断**。

    - ``pack_bin_size`` 仅表示 ``max_seq_length``：编码后 **长度超过则跳过该条**（不截断、不拆对话）。
    - 单条样本内通常 **不含** 预训练式 ``<|endoftext|>`` 合包分隔；``TrainDataCollator`` 仍按单段对话的 prefix/causal 规则构造掩码。
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
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = int(pack_bin_size)
        self.jsonl_path = str(jsonl_path)

        self.bos_enc_id = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.eos_enc_id = tokenizer.encode(f"{tokenizer.eos_token}\n", add_special_tokens=False)

        self.add_system_ratio = 0.2

        logger.info(
            "SFTDataset(one-row-one-sample, no pack/truncate): path=%s max_seq_len=%s",
            self.jsonl_path,
            self.max_seq_len,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shard_id, num_shards = PreTrainDataset._shard_info()
        seen = 0
        skipped_empty = 0
        skipped_long = 0

        for i, row in enumerate(_iter_jsonl_objects(self.jsonl_path)):
            if (i % num_shards) != shard_id:
                continue
            conversations = row.get("conversations", [])
            if not conversations:
                skipped_empty += 1
                continue
            enc = self._encode_conversation(conversations)
            if enc is None:
                skipped_empty += 1
                continue
            input_ids, labels = enc
            if not input_ids:
                skipped_empty += 1
                continue
            if len(input_ids) > self.max_seq_len:
                skipped_long += 1
                if skipped_long <= 3:
                    logger.warning(
                        "跳过超长样本（len=%s > max_seq_len=%s），不截断；行序≈%s",
                        len(input_ids),
                        self.max_seq_len,
                        i,
                    )
                continue

            seen += 1
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        logger.info(
            "SFTDataset shard=%s/%s emitted=%s skipped_empty=%s skipped_long=%s",
            shard_id,
            num_shards,
            seen,
            skipped_empty,
            skipped_long,
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
