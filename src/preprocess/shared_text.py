"""预训练 / SFT 共用的文本质量与拒绝回复检测（策略合并）。"""

from __future__ import annotations

import re
from typing import Iterable

_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_PUNCT_LIKE = set(
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。、；：？！「」『』（）【】《》…—·'
)

DEFAULT_REFUSE_SUBSTRINGS: tuple[str, ...] = (
    "我无法回答",
    "我不能回答",
    "作为人工智能",
    "由于系统限制",
    "抱歉，我无法",
    "I cannot answer",
    "I can't answer",
    "I'm unable to",
    "I am unable to",
    "as an AI language model",
    "As an AI",
)


def punctuation_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = sum(1 for c in s if c in _PUNCT_LIKE or (not c.isalnum() and not c.isspace()))
    return n / len(s)


def non_printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    bad = len(_NON_PRINTABLE_RE.findall(s))
    return bad / max(len(s), 1)


def normalize_special_markers(text: str, replacements: dict[str, str] | None) -> str:
    """将别名统一为规范串（如多种 think 标记合并）。"""
    if not replacements:
        return text
    out = text
    for src, dst in replacements.items():
        if src:
            out = out.replace(src, dst)
    return out


def looks_like_refuse_reply(text: str, extra_substrings: Iterable[str] | None = None) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for s in DEFAULT_REFUSE_SUBSTRINGS:
        if s.lower() in t:
            return True
    if extra_substrings:
        for s in extra_substrings:
            if s and str(s).lower() in t:
                return True
    return False
