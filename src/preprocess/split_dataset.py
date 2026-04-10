from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from src.preprocess.sft_conversation import count_turns


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        for row in rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_pretrain_train_val(
    rows: list[dict[str, Any]],
    val_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(rows)
    k = min(max(int(val_size), 0), n)
    if k <= 0:
        return rows, []
    rng = random.Random(seed)
    val_idx = set(rng.sample(range(n), k))
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i in val_idx:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def _has_tool_calls(row: dict[str, Any], conversations_field: str) -> bool:
    conv = row.get(conversations_field)
    if not isinstance(conv, list):
        return False
    for msg in conv:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tc = msg.get("tool_calls")
        if isinstance(tc, list) and tc:
            return True
        if isinstance(tc, str) and tc.strip():
            return True
    return False


def _is_multi_turn(row: dict[str, Any], conversations_field: str) -> bool:
    conv = row.get(conversations_field)
    if not isinstance(conv, list):
        return False
    return count_turns(conv) >= 2


def split_sft_train_and_eval_sets(
    rows: list[dict[str, Any]],
    tool_val_size: int,
    multi_turn_val_size: int,
    seed: int,
    conversations_field: str = "conversations",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(rows)
    if n <= 0:
        return [], [], []

    rng = random.Random(seed)
    all_indices = list(range(n))

    tool_candidates = [i for i in all_indices if _has_tool_calls(rows[i], conversations_field)]
    tool_k = min(max(int(tool_val_size), 0), len(tool_candidates))
    tool_idx = set(rng.sample(tool_candidates, tool_k)) if tool_k > 0 else set()

    remaining = [i for i in all_indices if i not in tool_idx]
    multi_candidates = [i for i in remaining if _is_multi_turn(rows[i], conversations_field)]
    multi_k = min(max(int(multi_turn_val_size), 0), len(multi_candidates))
    multi_idx = set(rng.sample(multi_candidates, multi_k)) if multi_k > 0 else set()

    train_rows: list[dict[str, Any]] = []
    tool_val_rows: list[dict[str, Any]] = []
    multi_turn_val_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i in tool_idx:
            tool_val_rows.append(row)
        elif i in multi_idx:
            multi_turn_val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, tool_val_rows, multi_turn_val_rows
