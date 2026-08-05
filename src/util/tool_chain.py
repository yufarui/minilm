"""Tool-call conversation chain helpers shared by preprocess and SFT training."""

from __future__ import annotations

from typing import Any


def has_orphan_tool_messages(messages: list[Any]) -> bool:
    """Return True when a ``tool``/``function`` turn lacks a prior assistant ``tool_calls`` list.

    Preprocess may delete unrepaired ``tool_calls`` while leaving following tool
    observations in place; training would then supervise answers after a
    fabricated ``<tool_response>`` with no ``<tool_call>``.
    """
    expect_tool = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            tcalls = msg.get("tool_calls")
            expect_tool = isinstance(tcalls, list) and len(tcalls) > 0
        elif role in ("tool", "function"):
            if not expect_tool:
                return True
        elif role == "user":
            expect_tool = False
    return False
