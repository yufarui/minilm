"""fasttext 语言识别模型（按解析后的权重路径单例）。"""

from __future__ import annotations

from pathlib import Path

from src.ref_model.registry import get_or_create
from src.util.path_util import resolve_under_project


def get_fasttext_model(model_path: str | Path):
    try:
        import fasttext  # type: ignore
    except ImportError as e:
        raise ImportError("lang_backend=fasttext 需要安装 fasttext：pip install fasttext") from e

    resolved = resolve_under_project(Path(model_path)).resolve()
    key = ("fasttext", str(resolved))

    def load():
        return fasttext.load_model(str(resolved))

    return get_or_create(key, load)
