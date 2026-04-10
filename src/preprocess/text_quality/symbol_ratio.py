"""不可见字符与标点占比异常检测（比例实现见 ``shared_text``）。"""

from __future__ import annotations

from dataclasses import dataclass

from src.preprocess.shared_text import non_printable_ratio, punctuation_ratio


@dataclass
class SymbolRatioBoundsConfig:
    max_non_printable_ratio: float = 0.05
    max_punctuation_ratio: float = 0.55


def passes_symbol_ratio_checks(text: str, cfg: SymbolRatioBoundsConfig) -> bool:
    if cfg.max_non_printable_ratio < 1.0 and non_printable_ratio(text) > cfg.max_non_printable_ratio:
        return False
    if punctuation_ratio(text) > cfg.max_punctuation_ratio:
        return False
    return True
