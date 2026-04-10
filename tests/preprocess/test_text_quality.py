from collections import Counter

import pytest

from src.preprocess.text_quality.language import (
    is_language_allowed,
    lang_matches,
    resolve_allowed_langs,
)
from src.preprocess.text_quality.length import LengthBoundsConfig, apply_char_length_bounds
from src.preprocess.text_quality.normalize import normalize_text
from src.preprocess.text_quality.pipeline import (
    REJECT_EMPTY,
    REJECT_LANGUAGE,
    REJECT_LENGTH,
    REJECT_NON_PRINTABLE,
    REJECT_PUNCTUATION,
    REJECT_TOKENS,
    apply_text_quality,
)
from src.preprocess.text_quality.symbol_ratio import SymbolRatioBoundsConfig, passes_symbol_ratio_checks
from src.preprocess.text_quality.tokens import (
    TokenBoundsConfig,
    needs_tokenizer,
    passes_token_bounds,
    top_token_entries,
)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [i for i, _ in enumerate(str(text).split(), start=1)]

    def convert_ids_to_tokens(self, ids):
        return [f"T{tid}" for tid in ids]


def _default_args():
    return {
        "length": LengthBoundsConfig(min_chars=5, max_chars=30, truncate=True),
        "symbols": SymbolRatioBoundsConfig(max_non_printable_ratio=0.2, max_punctuation_ratio=0.6),
        "allowed_langs": [],
        "lang_backend": "fasttext",
        "fasttext_model_path": None,
        "tokens": TokenBoundsConfig(min_tokens=None, max_tokens=None),
        "tokenizer": None,
    }


def test_normalize_text_handles_nfkc_and_newlines():
    raw = "ＡＢＣ\r\nline2  \rline3\t  "
    assert normalize_text(raw) == "ABC\nline2\nline3"


def test_apply_char_length_bounds_reject_and_truncate():
    cfg = LengthBoundsConfig(min_chars=3, max_chars=5, truncate=True)
    assert apply_char_length_bounds("ab", cfg) is None
    assert apply_char_length_bounds("abcdef", cfg) == "abcde"


def test_symbol_ratio_checks_for_non_printable_and_punctuation():
    strict_np = SymbolRatioBoundsConfig(max_non_printable_ratio=0.05, max_punctuation_ratio=1.0)
    strict_p = SymbolRatioBoundsConfig(max_non_printable_ratio=1.0, max_punctuation_ratio=0.2)
    assert not passes_symbol_ratio_checks("abc\x01def", strict_np)
    assert not passes_symbol_ratio_checks("!!!???...", strict_p)
    assert passes_symbol_ratio_checks("normal text content", strict_p)


def test_language_helpers_and_backend_validation(monkeypatch):
    assert lang_matches("zh-cn", "zh-tw")
    assert lang_matches("en", "en-US")
    assert resolve_allowed_langs(None) == ["zh-cn", "en"]
    assert resolve_allowed_langs([]) == []

    def _fake_predict(model_path, text, k=2, min_confidence=0.0):
        return ["en", "fr"]

    monkeypatch.setattr("src.preprocess.text_quality.language.predict_lang_fasttext", _fake_predict)
    assert is_language_allowed("hello", ["en"], "fasttext", "mock.bin", min_lang_confidence=0.2)
    assert not is_language_allowed("hello", ["zh"], "fasttext", "mock.bin", min_lang_confidence=0.2)

    with pytest.raises(ValueError):
        is_language_allowed("hello", ["en"], "langdetect", "mock.bin")


def test_token_helpers_and_top_entries():
    tok = DummyTokenizer()
    assert not needs_tokenizer(TokenBoundsConfig())
    assert needs_tokenizer(TokenBoundsConfig(min_tokens=2))
    assert passes_token_bounds("a b c", tok, TokenBoundsConfig(min_tokens=2, max_tokens=5))
    assert not passes_token_bounds("a", tok, TokenBoundsConfig(min_tokens=2, max_tokens=5))
    assert not passes_token_bounds("a b c d e f", tok, TokenBoundsConfig(min_tokens=2, max_tokens=5))

    top = top_token_entries(Counter({3: 5, 1: 2}), tok, k=2)
    assert top[0] == {"id": 3, "count": 5, "piece": "T3"}


def test_apply_text_quality_reject_reasons_cover_function_points():
    args = _default_args()

    r = apply_text_quality("   ", **args)
    assert r.text is None and r.reject == REJECT_EMPTY

    r = apply_text_quality("abc", **args)
    assert r.text is None and r.reject == REJECT_LENGTH

    args_np = _default_args()
    args_np["symbols"] = SymbolRatioBoundsConfig(max_non_printable_ratio=0.01, max_punctuation_ratio=1.0)
    r = apply_text_quality("hello\x01world!", **args_np)
    assert r.text is None and r.reject == REJECT_NON_PRINTABLE

    args_p = _default_args()
    args_p["symbols"] = SymbolRatioBoundsConfig(max_non_printable_ratio=1.0, max_punctuation_ratio=0.1)
    r = apply_text_quality("!!!!!!hello??????", **args_p)
    assert r.text is None and r.reject == REJECT_PUNCTUATION

    args_lang = _default_args()
    args_lang["allowed_langs"] = ["en"]
    args_lang["fasttext_model_path"] = None
    with pytest.raises(ValueError):
        apply_text_quality("this is valid length text", **args_lang)

    args_tok = _default_args()
    args_tok["tokens"] = TokenBoundsConfig(min_tokens=3, max_tokens=6)
    args_tok["tokenizer"] = DummyTokenizer()
    r = apply_text_quality("one two", **args_tok)
    assert r.text is None and r.reject == REJECT_TOKENS

    args_ok = _default_args()
    r = apply_text_quality("this is a valid clean sample", **args_ok)
    assert r.reject is None
    assert r.text == "this is a valid clean sample"

    args_lang_reject = _default_args()
    args_lang_reject["allowed_langs"] = ["en"]
    args_lang_reject["fasttext_model_path"] = "mock.bin"

    def _always_fr(model_path, text, k=2, min_confidence=0.0):
        return ["fr"]

    import src.preprocess.text_quality.language as lang_mod

    old = lang_mod.predict_lang_fasttext
    try:
        lang_mod.predict_lang_fasttext = _always_fr
        r = apply_text_quality("this should fail language gate", **args_lang_reject)
        assert r.text is None and r.reject == REJECT_LANGUAGE
    finally:
        lang_mod.predict_lang_fasttext = old
