"""MLM（如 BERT）：Tokenizer + ``AutoModelForMaskedLM`` 单例，按 (hub_id, device) 缓存。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.ref_model.registry import get_or_create


@dataclass
class MaskedLMReference:
    tokenizer: object
    model: object


def _resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def get_masked_lm_reference(model_name: str, device: str | None) -> MaskedLMReference:
    dev = _resolve_device(device)
    key = ("masked_lm", model_name, dev)

    def load() -> MaskedLMReference:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(dev)
        model.eval()
        return MaskedLMReference(tokenizer=tok, model=model)

    return get_or_create(key, load)
