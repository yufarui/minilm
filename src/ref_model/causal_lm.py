"""因果语言模型（PPL 等）：Tokenizer + ``AutoModelForCausalLM`` 单例缓存。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.ref_model.modelscope_hub import resolve_model_dir
from src.ref_model.registry import get_or_create


@dataclass
class CausalLMReference:
    tokenizer: object
    model: object


def _resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def get_causal_lm_reference(model_name: str, device: str | None) -> CausalLMReference:
    dev = _resolve_device(device)
    load_dir = resolve_model_dir(model_name)
    key = ("causal_lm", load_dir, dev)

    def load() -> CausalLMReference:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(load_dir, local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(load_dir, local_files_only=True)
        except OSError as e:
            raise RuntimeError(
                f"无法加载 PPL 参考模型目录 {load_dir!r}。"
                "请确认目录含完整 transformers 权重文件（config.json、tokenizer、model）。"
            ) from e
        model.to(dev)
        model.eval()
        return CausalLMReference(tokenizer=tok, model=model)

    return get_or_create(key, load)
