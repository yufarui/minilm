"""语种预测与白名单匹配（仅 fasttext）。"""

from __future__ import annotations

DEFAULT_ALLOWED_LANGS: tuple[str, ...] = ("zh-cn", "en")


def lang_matches(want: str, got: str) -> bool:
    a = want.lower().replace("_", "-")
    b = got.lower().replace("_", "-")
    if a == b:
        return True
    if a.startswith("zh") and b.startswith("zh"):
        return True
    if len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]:
        return True
    return False

def predict_lang_fasttext(
    model_path: str,
    text: str,
    *,
    k: int = 2,
    min_confidence: float = 0.0,
) -> list[str]:
    from src.ref_model import get_fasttext_model

    m = get_fasttext_model(model_path)
    t = text.replace("\n", " ")[:5000]
    try:
        labels, scores = m.predict(t, k=max(int(k), 1))
    except ValueError as e:
        msg = str(e)
        if "Unable to avoid copy while creating an array as requested" in msg:
            raise RuntimeError(
                "Detected fasttext + NumPy 2.x incompatibility. "
                "Please install NumPy < 2 in this environment, e.g. "
                "`uv pip install \"numpy<2\"` or `pip install \"numpy<2\"`."
            ) from e
        raise
    out: list[str] = []
    thr = max(float(min_confidence), 0.0)
    for lb, sc in zip(labels, scores, strict=False):
        conf = float(sc)
        if conf < thr:
            continue
        out.append(str(lb).replace("__label__", ""))
    return out


def resolve_allowed_langs(allowed_langs: list[str] | None) -> list[str]:
    if allowed_langs is not None:
        return list(allowed_langs)
    return list(DEFAULT_ALLOWED_LANGS)


def _matches_any_allowed(candidates: list[str], allowed: list[str]) -> bool:
    for cand in candidates:
        if any(lang_matches(want, cand) for want in allowed):
            return True
    return False


def is_language_allowed(
    text: str,
    allowed: list[str],
    backend: str,
    fasttext_model_path: str | None,
    min_lang_confidence: float = 0.0,
) -> bool:
    if not allowed:
        return True
    if backend != "fasttext":
        raise ValueError("仅支持 fasttext 语言检测，请将 lang_backend 设为 fasttext")
    if not fasttext_model_path:
        raise ValueError("lang_backend=fasttext 时必须设置 fasttext_model_path")
    cands = predict_lang_fasttext(
        fasttext_model_path,
        text,
        k=2,
        min_confidence=min_lang_confidence,
    )
    if not cands:
        return False
    return _matches_any_allowed(cands, allowed)

