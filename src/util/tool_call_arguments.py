"""Normalize ``tool_calls.arguments`` so chat templates emit valid JSON."""

from __future__ import annotations

import ast
import json
from typing import Any


def _literal_eval_structure(raw: str) -> dict[str, Any] | list[Any] | None:
    """Parse Python-literal dict/list (e.g. ``str(dict)`` single-quoted forms)."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if isinstance(value, (dict, list)):
        return value
    return None


def coerce_tool_call_arguments(arguments: Any) -> Any:
    """Return arguments safe to embed as the JSON ``arguments`` value.

    Chat ``chat_template.jinja`` emits string ``arguments`` raw. Empty strings,
    Python-repr dicts (``"{'k': 1}"``), and other non-JSON strings otherwise
    produce invalid ``<tool_call>`` JSON that SFT would supervise.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, (dict, list, bool, int, float)):
        return arguments
    if not isinstance(arguments, str):
        return {}

    text = arguments.strip()
    if not text:
        return {}

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        structured = _literal_eval_structure(text)
        return structured if structured is not None else {}

    # JSON string value must stay a quoted JSON string for the raw-emit path.
    if isinstance(parsed, str):
        return json.dumps(parsed, ensure_ascii=False)
    if parsed is None:
        return {}
    return parsed


def normalize_messages_tool_call_arguments(messages: list[dict[str, Any]]) -> int:
    """In-place coerce ``assistant.tool_calls[*].arguments``; return repair count."""
    repaired = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            targets = [tool_call]
            function = tool_call.get("function")
            if isinstance(function, dict):
                targets.append(function)
            for target in targets:
                if "arguments" not in target:
                    continue
                before = target["arguments"]
                after = coerce_tool_call_arguments(before)
                if after != before:
                    target["arguments"] = after
                    repaired += 1
    return repaired
