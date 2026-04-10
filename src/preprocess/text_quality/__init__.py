"""文本质量子模块：规范化、长度（含分布）、语言、符号比例、token（含分布）。"""

from __future__ import annotations

from src.preprocess.text_quality.language import (
    DEFAULT_ALLOWED_LANGS,
    lang_matches,
)
from src.preprocess.text_quality.length import equal_width_histogram
from src.preprocess.text_quality.normalize import normalize_text
from src.preprocess.text_quality.pipeline import (
    REJECT_EMPTY,
    REJECT_LANGUAGE,
    REJECT_LENGTH,
    REJECT_NON_PRINTABLE,
    REJECT_PUNCTUATION,
    REJECT_TOKENS,
    TextQualityResult,
    apply_text_quality,
)

__all__ = [
    "DEFAULT_ALLOWED_LANGS",
    "REJECT_EMPTY",
    "REJECT_LANGUAGE",
    "REJECT_LENGTH",
    "REJECT_NON_PRINTABLE",
    "REJECT_PUNCTUATION",
    "REJECT_TOKENS",
    "TextQualityResult",
    "apply_text_quality",
    "equal_width_histogram",
    "lang_matches",
    "normalize_text",
]
