"""Normalize ``system.tools`` list elements for chat-template rendering."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def coerce_tools_list_elements(tools: list[Any]) -> list[dict[str, Any]]:
    """Coerce a tools **list** into dict JSON schemas.

    ``transformers`` ``apply_chat_template`` rejects non-dict / non-callable
    tool entries before Jinja runs (``ValueError``). Some corpora store each
    schema as a JSON string inside the list; ``null`` / other junk also appears
    after messy exports. Parse recoverable string elements and drop the rest.
    """
    out: list[dict[str, Any]] = []
    for item in tools:
        if isinstance(item, dict):
            out.append(item)
            continue
        if isinstance(item, str):
            s = item.strip()
            if not s:
                continue
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as e:
                logger.warning("system.tools 列表元素 JSON 无效，已跳过: %s", e)
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
            else:
                logger.warning(
                    "system.tools 列表元素解析后不是对象（%s），已跳过",
                    type(parsed).__name__,
                )
            continue
        if item is None:
            continue
        logger.warning(
            "system.tools 列表元素类型无效（%s），已跳过",
            type(item).__name__,
        )
    return out
