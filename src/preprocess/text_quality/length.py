"""字符长度：上下界/截断检测与长度类数值的分布（直方图）。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LengthBoundsConfig:
    min_chars: int = 20
    max_chars: int | None = 200_000
    truncate: bool = True


def apply_char_length_bounds(text: str, cfg: LengthBoundsConfig) -> str | None:
    """
    对已规范化文本做长度处理。

    - 短于 ``min_chars``：返回 ``None``。
    - 超过 ``max_chars``：``truncate`` 为 True 时截断，否则 ``None``。
    """
    if len(text) < cfg.min_chars:
        return None
    if cfg.max_chars is not None and len(text) > cfg.max_chars:
        if not cfg.truncate:
            return None
        return text[: cfg.max_chars]
    return text


def equal_width_histogram(values: list[float], bins: int = 20) -> tuple[list[float], list[int]]:
    """等宽分桶（字符长度、token 长度等共用）；返回 (bin_edges, counts)。"""
    arr = [float(v) for v in values if not math.isnan(float(v)) and math.isfinite(float(v))]
    if not arr:
        return [], []
    vmin, vmax = min(arr), max(arr)
    if vmax <= vmin:
        return [vmin, vmax + 1.0], [len(arr)]
    step = (vmax - vmin) / float(max(bins, 1))
    edges = [vmin + i * step for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for v in arr:
        idx = int((v - vmin) / step)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return edges, counts
