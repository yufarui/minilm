"""sentence-transformers 编码器（主题审计等），按模型名/Hub id 单例。"""

from __future__ import annotations

from src.ref_model.modelscope_hub import resolve_model_dir
from src.ref_model.registry import get_or_create


def get_sentence_transformer(model_name: str):
    load_dir = resolve_model_dir(model_name)
    key = ("sentence_transformer", load_dir)

    def load():
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(load_dir, local_files_only=True)

    return get_or_create(key, load)
