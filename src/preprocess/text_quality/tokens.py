"""Token：基于 tokenizer 的条数上下界检测，以及诊断用分布（直方图、高频 token）。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.preprocess.text_quality.length import equal_width_histogram


@dataclass
class TokenBoundsConfig:
    min_tokens: int | None = None
    """``None`` 表示不做下限。"""
    max_tokens: int | None = None
    """``None`` 表示不做上限。"""


def tokenize_length(text: str, tokenizer: Any) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def needs_tokenizer(cfg: TokenBoundsConfig) -> bool:
    return cfg.min_tokens is not None or cfg.max_tokens is not None


def passes_token_bounds(text: str, tokenizer: Any, cfg: TokenBoundsConfig) -> bool:
    if not needs_tokenizer(cfg):
        return True
    n = tokenize_length(text, tokenizer)
    if cfg.min_tokens is not None and n < cfg.min_tokens:
        return False
    if cfg.max_tokens is not None and n > cfg.max_tokens:
        return False
    return True


def top_token_entries(
    token_counter: Counter[int],
    tokenizer: Any,
    k: int = 40,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tid, cnt in token_counter.most_common(k):
        piece = tokenizer.convert_ids_to_tokens([tid])[0]
        out.append({"id": tid, "count": cnt, "piece": piece})
    return out


def histogram_from_lengths(lengths: list[float], bins: int = 20) -> dict[str, Any]:
    e, c = equal_width_histogram(lengths, bins=bins)
    return {"edges": e, "counts": c}
