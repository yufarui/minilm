"""预训练 / SFT 共用的文本质量入口；具体步骤见 ``text_quality`` 子模块。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.preprocess.text_quality.language import DEFAULT_ALLOWED_LANGS, lang_matches
from src.preprocess.text_quality.length import LengthBoundsConfig
from src.preprocess.text_quality.normalize import normalize_text
from src.preprocess.text_quality.pipeline import (
    TextQualityResult,
    apply_text_quality,
)
from src.preprocess.text_quality.symbol_ratio import SymbolRatioBoundsConfig
from src.preprocess.text_quality.tokens import TokenBoundsConfig


@dataclass
class BasicCleanConfig:
    min_chars: int = 20
    max_chars: int | None = 200_000
    truncate: bool = True
    max_non_printable_ratio: float = 0.05
    max_punctuation_ratio: float = 0.55
    allowed_langs: list[str] | None = None
    """语言白名单（与 ``lang_matches`` 规则一致）。

    - ``None``：``DEFAULT_ALLOWED_LANGS``（默认中、英）。
    - ``[]``：不做语言过滤。
    """
    lang_backend: str = "fasttext"
    """仅支持 ``fasttext``。"""
    fasttext_model_path: str | None = None
    min_lang_confidence: float = 0.0
    """语种预测置信度下限（当前仅 fasttext 生效）。"""
    min_tokens: int | None = None
    max_tokens: int | None = None
    """非空时需配合调用方提供 tokenizer。"""


def _length_cfg(c: BasicCleanConfig) -> LengthBoundsConfig:
    return LengthBoundsConfig(
        min_chars=c.min_chars,
        max_chars=c.max_chars,
        truncate=c.truncate,
    )


def _symbol_cfg(c: BasicCleanConfig) -> SymbolRatioBoundsConfig:
    return SymbolRatioBoundsConfig(
        max_non_printable_ratio=c.max_non_printable_ratio,
        max_punctuation_ratio=c.max_punctuation_ratio,
    )


def _token_cfg(c: BasicCleanConfig) -> TokenBoundsConfig:
    return TokenBoundsConfig(min_tokens=c.min_tokens, max_tokens=c.max_tokens)


def apply_basic_text_quality(
    text: str,
    cfg: BasicCleanConfig,
    tokenizer: object | None = None,
) -> TextQualityResult:
    """与 ``basic_clean`` 相同逻辑，但返回结构化结果（便于统计剔除原因）。"""
    return apply_text_quality(
        text,
        length=_length_cfg(cfg),
        symbols=_symbol_cfg(cfg),
        allowed_langs=cfg.allowed_langs,
        lang_backend=cfg.lang_backend,
        fasttext_model_path=cfg.fasttext_model_path,
        min_lang_confidence=cfg.min_lang_confidence,
        tokens=_token_cfg(cfg),
        tokenizer=tokenizer,
    )


def basic_clean(text: str, cfg: BasicCleanConfig, tokenizer: object | None = None) -> str | None:
    """返回清洗后文本；不满足条件时返回 ``None``（丢弃）。"""
    return apply_basic_text_quality(text, cfg, tokenizer=tokenizer).text


def detect_lang(text: str) -> str | None:
    """兼容旧名：当前仅支持 fasttext，请改用 ``predict_lang_fasttext``。"""
    raise NotImplementedError("仅支持 fasttext，请使用 predict_lang_fasttext 并传入模型路径")


def dedupe_consecutive_paragraphs(text: str) -> str:
    """去除连续重复的段落（双换行分段）。"""
    paras = [p.strip() for p in text.split("\n\n")]
    out: list[str] = []
    prev_norm: str | None = None
    for p in paras:
        if not p:
            continue
        key = re.sub(r"\s+", " ", p)
        if prev_norm is not None and key == prev_norm:
            continue
        out.append(p)
        prev_norm = key
    return "\n\n".join(out)


__all__ = [
    "BasicCleanConfig",
    "DEFAULT_ALLOWED_LANGS",
    "TextQualityResult",
    "apply_basic_text_quality",
    "basic_clean",
    "dedupe_consecutive_paragraphs",
    "detect_lang",
    "lang_matches",
    "normalize_text",
]
