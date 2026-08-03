"""Normalize chat message ``content`` (OpenAI multipart → plain text)."""

from __future__ import annotations

from typing import Any


def coerce_content_to_text(content: Any) -> tuple[str | None, bool]:
    """将消息 ``content`` 规范为字符串。

    支持 OpenAI multipart 列表（``[{"type":"text","text":"..."}, ...]``）。
    返回 ``(text, ok)``：``ok=False`` 表示无法安全转换（应跳过样本）；
    ``content is None`` 视为空串且 ``ok=True``（tool_calls 轮常见）。
    """
    if content is None:
        return "", True
    if isinstance(content, str):
        return content, True
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append(part)
                continue
            if not isinstance(part, dict):
                return None, False
            text = part.get("text")
            if isinstance(text, str):
                if text:
                    parts.append(text)
                continue
            # 少数导出用 content 字段承载文本
            nested = part.get("content")
            if isinstance(nested, str):
                if nested:
                    parts.append(nested)
                continue
            # 跳过非文本 part（如图片）；纯非文本列表 → 空串
            ptype = part.get("type")
            if ptype in {None, "text"}:
                return None, False
        return "\n".join(parts), True
    return None, False


def normalize_messages_content(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, int]:
    """规范每条消息的 ``content`` 为字符串；失败返回 ``(None, 0)``。"""
    out: list[dict[str, Any]] = []
    coerced = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        mm = dict(m)
        if "content" in mm:
            raw = mm["content"]
            text, ok = coerce_content_to_text(raw)
            if not ok or text is None:
                return None, 0
            if not isinstance(raw, str) and raw is not None:
                coerced += 1
            mm["content"] = text
        out.append(mm)
    return out, coerced
