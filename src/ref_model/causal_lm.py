"""因果语言模型（PPL 等）：Tokenizer + ``AutoModelForCausalLM`` 单例，按 (hub_id, device) 缓存。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.ref_model.registry import get_or_create


@dataclass
class CausalLMReference:
    tokenizer: object
    model: object


def _resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def get_causal_lm_reference(model_name: str, device: str | None) -> CausalLMReference:
    dev = _resolve_device(device)
    key = ("causal_lm", model_name, dev)

    def load() -> CausalLMReference:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(dev)
        model.eval()
        return CausalLMReference(tokenizer=tok, model=model)

    return get_or_create(key, load)
