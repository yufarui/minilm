from __future__ import annotations

import copy
import json
import logging
import random
import re
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
    - **labels**：与 ``chat_template.jinja`` 对齐，用「模板字符串 + offset_mapping」标 assistant 段；避免子词边界导致 ``prompt[i:i+k]==encode(片段)`` 永远不匹配、进而全为 ``-100``、loss 为 nan。
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

        self.add_system_ratio = 0.2
        eos = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
        self._assistant_block = re.compile(
            rf"<\|im_start\|>assistant\n(.*?){re.escape(eos)}",
            re.DOTALL,
        )

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
        skipped_no_supervision = 0

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
            if not any(x != -100 for x in labels):
                skipped_no_supervision += 1
                if skipped_no_supervision <= 3:
                    logger.warning(
                        "跳过无监督 token（labels 全为 -100），行序≈%s",
                        i,
                    )
                continue

            seen += 1
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        logger.info(
            "SFTDataset shard=%s/%s emitted=%s skipped_empty=%s skipped_long=%s skipped_no_supervision=%s",
            shard_id,
            num_shards,
            seen,
            skipped_empty,
            skipped_long,
            skipped_no_supervision,
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

        text = self.tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            open_think=False,
        )
        if not isinstance(text, str):
            logger.warning(
                "apply_chat_template(tokenize=False) 期望 str，得到 %s，跳过该条",
                type(text).__name__,
            )
            return None

        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        labels = self._labels_from_template_offsets(text, input_ids, offsets)
        # 抽样检查正确性
        r = random.random()
        if r < 0.1:
            logger.info(f"text\n:{text},")
        return input_ids, labels

    def _labels_from_template_offsets(
        self,
        text: str,
        input_ids: list[int],
        offset_mapping: list[tuple[int, int]],
    ) -> list[int]:
        """按 chat 模板字符串中的 assistant 段（含多轮）映射到 token；``<|im_end|>`` 不计入 loss。"""
        labels = [-100] * len(input_ids)
        for m in self._assistant_block.finditer(text):
            c0, c1 = m.span(1)
            for ti, (a, b) in enumerate(offset_mapping):
                if a >= c1 or b <= c0:
                    continue
                if a < c1 and b > c0:
                    labels[ti] = input_ids[ti]
        return labels

if __name__ == "__main__":
    from src.ref_model.tokenizer_local import get_auto_tokenizer_local
    from src.util.path_util import resolve_under_project

    tok_path = resolve_under_project("tokenizer/minilm")
    tokenizer = get_auto_tokenizer_local(tok_path, trust_remote_code=True)
    train_dataset = SFTDataset(
        "data/stf/sft_train.jsonl",
        tokenizer,
        pack_bin_size=8192,
    )