"""串联规范化 → 长度 → 符号比例 → 语言 → token 边界。"""

from __future__ import annotations

from dataclasses import dataclass

from src.preprocess.text_quality.language import (
    is_language_allowed,
    resolve_allowed_langs,
)
from src.preprocess.text_quality.length import LengthBoundsConfig, apply_char_length_bounds
from src.preprocess.text_quality.normalize import normalize_text
from src.preprocess.text_quality.symbol_ratio import SymbolRatioBoundsConfig, passes_symbol_ratio_checks
from src.preprocess.text_quality.tokens import TokenBoundsConfig, needs_tokenizer, passes_token_bounds


# 与统计、日志对齐的拒绝原因
REJECT_EMPTY = "empty"
REJECT_LENGTH = "length"
REJECT_NON_PRINTABLE = "non_printable"
REJECT_PUNCTUATION = "punctuation"
REJECT_LANGUAGE = "language"
REJECT_TOKENS = "tokens"


@dataclass(frozen=True)
class TextQualityResult:
    text: str | None
    reject: str | None


def apply_text_quality(
    raw: str,
    *,
    length: LengthBoundsConfig,
    symbols: SymbolRatioBoundsConfig,
    allowed_langs: list[str] | None,
    lang_backend: str,
    fasttext_model_path: str | None,
    min_lang_confidence: float = 0.0,
    tokens: TokenBoundsConfig,
    tokenizer: object | None,
) -> TextQualityResult:
    """
    对原始字符串做完整文本质量链；返回清洗后文本或拒绝原因。

    ``tokenizer`` 在需要 token 界且 ``needs_tokenizer(tokens)`` 时必须非空。
    """
    if not raw or not str(raw).strip():
        return TextQualityResult(None, REJECT_EMPTY)

    t = normalize_text(str(raw))
    t = apply_char_length_bounds(t, length)
    if t is None:
        return TextQualityResult(None, REJECT_LENGTH)

    if not passes_symbol_ratio_checks(t, symbols):
        from src.preprocess.shared_text import non_printable_ratio, punctuation_ratio

        if symbols.max_non_printable_ratio < 1.0 and non_printable_ratio(t) > symbols.max_non_printable_ratio:
            return TextQualityResult(None, REJECT_NON_PRINTABLE)
        return TextQualityResult(None, REJECT_PUNCTUATION)

    allowed = resolve_allowed_langs(allowed_langs)
    try:
        if not is_language_allowed(
            t,
            allowed,
            lang_backend,
            fasttext_model_path,
            min_lang_confidence=min_lang_confidence,
        ):
            return TextQualityResult(None, REJECT_LANGUAGE)
    except ImportError:
        raise

    if needs_tokenizer(tokens):
        if tokenizer is None:
            raise ValueError("已设置 min_tokens/max_tokens 但未提供 tokenizer")
        if not passes_token_bounds(t, tokenizer, tokens):
            return TextQualityResult(None, REJECT_TOKENS)

    return TextQualityResult(t, None)
