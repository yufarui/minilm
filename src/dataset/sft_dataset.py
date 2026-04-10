from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """对话 SFT：JSONL 每行含 ``conversations``；仅用 ``assistant`` 段计算 loss。
    载入后按序列长度排序，first-fit packing 拼包（包内样本间用 ``<|endoftext|>`` 拼接，对应 label 为 -100）。
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
            pack_sort_order: str = "shortest_first",

    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pack_bin_size = int(pack_bin_size)
        self.pack_sort_order = pack_sort_order

        sep_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        self.jsonl_path = str(jsonl_path)
        self.samples = load_dataset("json", data_files=self.jsonl_path, split="train")

        self.bos_enc_id = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.eos_enc_id = tokenizer.encode(f"{tokenizer.eos_token}\n", add_special_tokens=False)

        self.add_system_ratio = 0.2

        doc_pairs: list[tuple[list[int], list[int]]] = []
        skipped_empty = 0
        truncated_docs = 0

        for i in range(len(self.samples)):
            sample = self.samples[i]
            conversations = sample.get("conversations", [])
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
            if len(input_ids) > self.pack_bin_size:
                input_ids = input_ids[: self.pack_bin_size]
                labels = labels[: self.pack_bin_size]
                truncated_docs += 1
            doc_pairs.append((input_ids, labels))

        if pack_sort_order == "longest_first":
            doc_pairs.sort(key=lambda p: len(p[0]), reverse=True)
        elif pack_sort_order == "shortest_first":
            doc_pairs.sort(key=lambda p: len(p[0]), reverse=False)

        packed = self._pack_sft_pairs_into_bins(doc_pairs, self.pack_bin_size, int(sep_id))

        self._packed: list[dict[str, torch.Tensor]] = []
        for input_ids, labels in packed:
            self._packed.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

        logger.info(
            "SFTDataset packing: raw_convs=%s bins=%s pack_bin_size=%s sort=%s "
            "skipped_empty=%s truncated_to_bin=%s",
            len(doc_pairs),
            len(self._packed),
            self.pack_bin_size,
            pack_sort_order,
            skipped_empty,
            truncated_docs,
        )

    def __len__(self) -> int:
        return len(self._packed)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._packed[index]

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

        prompt = self.tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=False,
            tools=tools,
            # 关闭 open_think
            open_think=False,
        )
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        prompt = list(prompt)

        labels = self.generate_labels(prompt)
        return prompt, labels

    def generate_labels(self, prompt: list[int]) -> list[int]:
        labels = [-100] * len(prompt)
        i = 0
        bos, eos = self.bos_enc_id, self.eos_enc_id
        lbos, leos = len(bos), len(eos)
        while i < len(prompt):
            if prompt[i: i + lbos] == bos:
                start = i + lbos
                j = start
                while j < len(prompt):
                    labels[j] = prompt[j]
                    if prompt[j: j + leos] == eos:
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
