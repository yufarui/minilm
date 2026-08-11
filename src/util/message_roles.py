"""Normalize chat roles / reasoning fields before chat_template render."""

from __future__ import annotations

from typing import Any


def normalize_developer_roles(messages: list[dict[str, Any]]) -> int:
    """Map OpenAI ``developer`` roles to ``system`` (in place).

    ``chat_template.jinja`` only emits ``system`` / ``user`` / ``assistant`` / ``tool``.
    With default ``strict_role_order: false``, ``developer`` rows are kept by preprocess
    but silently omitted from the rendered prompt — policy/instruction text disappears
    while dependent assistant answers remain supervised.

    Returns the number of roles rewritten.
    """
    n = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "developer":
            msg["role"] = "system"
            n += 1
    return n


def materialize_reasoning_content(
    messages: list[dict[str, Any]],
    *,
    open_think: bool = False,
) -> int:
    """When not training think spans, fold orphan ``reasoning_content`` into ``content``.

    SFT/DPO/tokenizer paths hardcode ``open_think=False``. The template then discards
    ``reasoning_content`` and emits only ``content``. Rows that store the answer solely
    in ``reasoning_content`` (empty/missing ``content``) become empty assistant turns
    supervised as immediate ``<|im_end|>``.

    When ``open_think`` is True, leave fields untouched (template emits ``<think>``).

    Returns how many assistant messages were rewritten.
    """
    if open_think:
        return 0
    n = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        reasoning = msg.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            # Final answer already present; drop unused reasoning for the non-think path.
            msg.pop("reasoning_content", None)
            continue
        msg["content"] = reasoning
        msg.pop("reasoning_content", None)
        n += 1
    return n
