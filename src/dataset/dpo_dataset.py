from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class DPODataset:
    """DPO 偏好数据。

    支持两种 JSONL 行格式：

    1. 对话偏好（与 TRL 对齐）：仅 ``chosen`` / ``rejected``，值为消息列表
       ``[{"role": "user"|"assistant", "content": "..."}, ...]``，最后一条须为
       ``assistant``；二者除最后一条 assistant 外应一致。此时须提供 ``tokenizer``，
       用 ``apply_chat_template(..., add_generation_prompt=True)`` 生成 ``prompt``。

    2. 扁平格式：``prompt`` / ``chosen`` / ``rejected`` 均为字符串（``tokenizer`` 可省略）。
    """

    OUTPUT_COLUMNS = frozenset({"prompt", "chosen", "rejected"})
    MIN_CHAT_COLUMNS = frozenset({"chosen", "rejected"})

    def __init__(self, json_path: str | Path, tokenizer: PreTrainedTokenizerBase | None = None) -> None:
        self.json_path = str(json_path)
        self.tokenizer = tokenizer
        self.dataset = self._load(self.json_path, tokenizer)
        logger.info("DPODataset loaded: %s samples from %s", len(self.dataset), self.json_path)

    @staticmethod
    def _is_chat_messages(value: Any) -> bool:
        if not isinstance(value, list) or not value:
            return False
        first = value[0]
        if not isinstance(first, Mapping):
            return False
        return "role" in first and "content" in first

    @staticmethod
    def _tools_from_messages(messages: list[dict[str, Any]]) -> list[Any] | None:
        if not messages or messages[0].get("role") != "system":
            return None
        raw = messages[0].get("tools")
        if raw is None:
            return None
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return None
            try:
                return json.loads(s)
            except json.JSONDecodeError as e:
                logger.warning("system.tools JSON 无效，将不传 tools: %s", e)
                return None
        return None

    @classmethod
    def _chat_triplet(
        cls,
        chosen: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, str]:
        if not cls._is_chat_messages(chosen) or not cls._is_chat_messages(rejected):
            raise ValueError("chosen/rejected 须为非空消息列表，且元素含 role/content。")
        if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
            raise ValueError("chosen 与 rejected 的最后一条须为 role=assistant。")

        prefix_c = [dict(m) for m in chosen[:-1]]
        prefix_r = [dict(m) for m in rejected[:-1]]
        if len(prefix_c) != len(prefix_r):
            logger.warning(
                "chosen/rejected 前缀轮数不一致 (chosen=%s rejected=%s)，以 chosen 前缀生成 prompt。",
                len(prefix_c),
                len(prefix_r),
            )
        else:
            for i, (a, b) in enumerate(zip(prefix_c, prefix_r, strict=True)):
                if a.get("role") != b.get("role") or str(a.get("content", "")) != str(b.get("content", "")):
                    logger.warning(
                        "chosen/rejected 在第 %s 轮前缀不一致，以 chosen 前缀生成 prompt。", i
                    )
                    break

        tools = cls._tools_from_messages(prefix_c)
        prompt = tokenizer.apply_chat_template(
            prefix_c,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            open_think=False,
        )
        if not isinstance(prompt, str):
            prompt = tokenizer.decode(prompt) if hasattr(prompt, "tolist") else str(prompt)

        return {
            "prompt": prompt,
            "chosen": str(chosen[-1]["content"]),
            "rejected": str(rejected[-1]["content"]),
        }

    @classmethod
    def _load(cls, json_path: str, tokenizer: PreTrainedTokenizerBase | None) -> Dataset:
        ds = load_dataset("json", data_files=json_path, split="train")
        names = set(ds.column_names)
        if not cls.MIN_CHAT_COLUMNS <= names:
            raise ValueError(
                f"DPO 数据至少需字段 chosen/rejected，当前: {ds.column_names}。"
            )

        peek = ds[0]
        use_chat = cls._is_chat_messages(peek["chosen"]) and cls._is_chat_messages(peek["rejected"])
        if use_chat and tokenizer is None:
            raise ValueError(
                "数据为对话格式（chosen/rejected 为消息列表）时，须在 DPODataset(..., tokenizer=...) 传入 tokenizer。"
            )
        if not use_chat and "prompt" not in names:
            raise ValueError(
                "扁平格式需同时提供 prompt/chosen/rejected 三个字符串字段。"
            )

        def _row_to_trl(sample: dict[str, Any]) -> dict[str, str]:
            c_raw, r_raw = sample["chosen"], sample["rejected"]
            if cls._is_chat_messages(c_raw) and cls._is_chat_messages(r_raw):
                assert tokenizer is not None
                return cls._chat_triplet(list(c_raw), list(r_raw), tokenizer)
            if cls._is_chat_messages(c_raw) ^ cls._is_chat_messages(r_raw):
                raise ValueError("同一条样本中 chosen 与 rejected 须同为消息列表或同为字符串。")
            if "prompt" not in sample:
                raise ValueError("非对话格式时缺少 prompt。")
            return {
                "prompt": str(sample["prompt"]),
                "chosen": str(c_raw),
                "rejected": str(r_raw),
            }

        drop = [c for c in ds.column_names if c not in cls.OUTPUT_COLUMNS]
        return ds.map(_row_to_trl, remove_columns=drop)

    def as_hf_dataset(self) -> Dataset:
        return self.dataset
