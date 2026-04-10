"""SFT 多轮对话：角色链校验、tool_calls JSON 修复、文本拼接。"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def conversation_concat_text(messages: list[dict[str, Any]]) -> str:
    """用于语言检测、近似去重、长度与符号比例。"""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if c is not None:
            parts.append(str(c))
        tc = m.get("tool_calls")
        if isinstance(tc, str) and tc.strip():
            parts.append(tc)
        elif isinstance(tc, list):
            parts.append(json.dumps(tc, ensure_ascii=False))
    return "\n".join(parts)


def validate_role_chain(messages: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """
    期望：可选 ``system`` 后，重复 ``user`` → ``assistant``；
    若 ``assistant`` 含非空 ``tool_calls``，则须紧跟若干 ``tool``，再跟一个 ``assistant``。
    """
    if not messages:
        return False, "empty"
    n = len(messages)
    i = 0
    if messages[0].get("role") == "system":
        i = 1
        if i >= n or messages[i].get("role") != "user":
            return False, "after_system_expect_user"
    while i < n:
        if messages[i].get("role") != "user":
            return False, f"expect_user_at_{i}"
        i += 1
        if i >= n or messages[i].get("role") != "assistant":
            return False, f"expect_assistant_at_{i}"
        asst = messages[i]
        i += 1
        tcalls = asst.get("tool_calls")
        has_tools = isinstance(tcalls, list) and len(tcalls) > 0
        if has_tools:
            saw_tool = False
            while i < n and messages[i].get("role") == "tool":
                saw_tool = True
                i += 1
            if not saw_tool:
                return False, "tool_calls_without_tool_messages"
            if i >= n or messages[i].get("role") != "assistant":
                return False, "expect_assistant_after_tool"
            i += 1
    return True, None


def _strip_json_fences(s: str) -> str:
    t = s.strip()
    t = _TOOL_FENCE_RE.sub("", t)
    return t.strip()


def _repair_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[\]}])", r"\1", s)


def try_repair_tool_calls_json(raw: str) -> tuple[Any | None, bool]:
    """
    尝试将 ``tool_calls`` 字符串解析为 JSON（列表或对象）。
    返回 (解析结果, 是否成功)；失败时结果为 None。
    """
    s = _strip_json_fences(raw.strip())
    if not s:
        return None, False
    for _ in range(4):
        try:
            return json.loads(s), True
        except json.JSONDecodeError:
            s2 = _repair_trailing_commas(s)
            if s2 == s:
                break
            s = s2
    # 括号补齐（仅当明显缺右括号）
    open_b = s.count("[")
    close_b = s.count("]")
    open_c = s.count("{")
    close_c = s.count("}")
    s2 = s
    if open_b > close_b:
        s2 += "]" * (open_b - close_b)
    if open_c > close_c:
        s2 += "}" * (open_c - close_c)
    if s2 != s:
        try:
            return json.loads(s2), True
        except json.JSONDecodeError:
            pass
    return None, False


def normalize_messages_tool_calls(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    将 ``assistant.tool_calls`` 从字符串尽量解析为列表；无法解析则移除该键以免下游模板崩溃。
    返回 (新消息列表, 成功修复条数)。
    """
    out = []
    repaired = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        mm = dict(m)
        if mm.get("role") == "assistant" and "tool_calls" in mm:
            raw = mm["tool_calls"]
            if isinstance(raw, str):
                parsed, ok = try_repair_tool_calls_json(raw)
                if ok and parsed is not None:
                    mm["tool_calls"] = parsed
                    repaired += 1
                else:
                    del mm["tool_calls"]
            elif isinstance(raw, list):
                pass
            else:
                del mm["tool_calls"]
        out.append(mm)
    return out, repaired


def assistant_contents(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(m.get("content") or "")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]


def tool_calls_json_length(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tc = m.get("tool_calls")
        if isinstance(tc, str):
            total += len(tc)
        elif isinstance(tc, list):
            total += len(json.dumps(tc, ensure_ascii=False))
    return total


def count_turns(messages: list[dict[str, Any]]) -> int:
    """用户轮数（``user`` 消息个数）。"""
    return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")
