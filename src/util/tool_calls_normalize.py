"""Normalize assistant.tool_calls list elements for chat-template rendering."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_json_fences(s: str) -> str:
    t = s.strip()
    t = _TOOL_FENCE_RE.sub("", t)
    return t.strip()


def _repair_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[\]}])", r"\1", s)


def try_repair_tool_calls_json(raw: str) -> tuple[Any | None, bool]:
    """Try to parse a tool_calls JSON string (list or object)."""
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


def normalize_tool_call_item(item: Any) -> dict[str, Any] | None:
    """Normalize one ``tool_calls`` list element to a dict.

    Accepts a dict, or a JSON object string. Also parses stringified OpenAI
    ``function`` payloads. Returns ``None`` if the element cannot be used.
    """
    if isinstance(item, str):
        parsed, ok = try_repair_tool_calls_json(item)
        if not ok or not isinstance(parsed, dict):
            return None
        item = parsed
    if not isinstance(item, dict):
        return None
    fn = item.get("function")
    if isinstance(fn, str):
        parsed_fn, ok = try_repair_tool_calls_json(fn)
        if not ok or not isinstance(parsed_fn, dict):
            return None
        item = dict(item)
        item["function"] = parsed_fn
    return item


def normalize_tool_calls_list(raw: list[Any]) -> list[dict[str, Any]] | None:
    """Normalize a ``tool_calls`` list to dict elements; ``None`` if any element fails."""
    out: list[dict[str, Any]] = []
    for item in raw:
        norm = normalize_tool_call_item(item)
        if norm is None:
            return None
        out.append(norm)
    return out
