"""数据预处理等场景共用的参考模型：进程内单例懒加载。"""

from __future__ import annotations

from src.ref_model.causal_lm import CausalLMReference, get_causal_lm_reference
from src.ref_model.fasttext_model import get_fasttext_model
from src.ref_model.masked_lm import MaskedLMReference, get_masked_lm_reference
from src.ref_model.registry import clear_ref_model_cache
from src.ref_model.sentence_transformer import get_sentence_transformer
from src.ref_model.tokenizer_local import get_auto_tokenizer_local

__all__ = [
    "CausalLMReference",
    "MaskedLMReference",
    "clear_ref_model_cache",
    "get_auto_tokenizer_local",
    "get_causal_lm_reference",
    "get_fasttext_model",
    "get_masked_lm_reference",
    "get_sentence_transformer",
]
