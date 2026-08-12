"""Normalize non-string chat ``content`` before ``chat_template.jinja``.

The template only keeps ``content`` when it ``is string``; otherwise it sets
``content = ''``. Structured tool results (``dict``) and scalar payloads are
common in exports and must be serialized, or tool observations / answers are
silently wiped while dependent assistant turns remain supervised.

OpenAI multipart lists (``[{"type":"text","text":...}, ...]``) are intentionally
left unchanged here — that path is covered by a separate fix.
"""

from __future__ import annotations

import json
from typing import Any


def coerce_content_to_text(content: Any) -> tuple[str | None, bool]:
    """将非 list 的 ``content`` 规范为字符串。

    返回 ``(text, ok)``：
    - ``ok=True``：已得到可写入模板的字符串（``None`` content → ``""``）
    - ``ok=False``：无法安全转换（调用方应跳过样本）
    - 对于 ``list``：返回 ``(None, True)`` 且约定调用方**不要改写**该字段
      （multipart 由其它修复处理；此处不 json.dumps，以免把 parts 结构训进模型）
    """
    if content is None:
        return "", True
    if isinstance(content, str):
        return content, True
    if isinstance(content, list):
        # 留给 multipart 专用路径；normalize 侧保持原值。
        return None, True
    if isinstance(content, (dict, int, float, bool)):
        try:
            return json.dumps(content, ensure_ascii=False), True
        except (TypeError, ValueError):
            return None, False
    return None, False


def normalize_messages_content(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, int]:
    """将 dict/scalar ``content`` 序列化为字符串；失败返回 ``(None, 0)``。

    ``list`` content（含 OpenAI multipart）原样保留。
    """
    out: list[dict[str, Any]] = []
    coerced = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        mm = dict(m)
        if "content" in mm:
            raw = mm["content"]
            if isinstance(raw, list):
                out.append(mm)
                continue
            text, ok = coerce_content_to_text(raw)
            if not ok or text is None:
                return None, 0
            if not isinstance(raw, str) and raw is not None:
                coerced += 1
            mm["content"] = text
        out.append(mm)
    return out, coerced
