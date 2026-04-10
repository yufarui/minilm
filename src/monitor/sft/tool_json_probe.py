"""定期用前缀生成文本，从输出中抽取 JSON 子串并尝试 ``json.loads``，估计可解析比例。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _first_balanced_brace_chunk(s: str) -> str | None:
    """从首个 ``{`` 起做括号深度匹配（忽略字符串内引号简化为常见 JSON 探针）。"""
    i = s.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
    return None


def _render_generation_prompt(tokenizer: Any, prompt_item: Any) -> str:
    if isinstance(prompt_item, str):
        return prompt_item
    conv = None
    tools = None
    if isinstance(prompt_item, list):
        conv = prompt_item
    elif isinstance(prompt_item, dict):
        conv = prompt_item.get("conversations")
        tools = prompt_item.get("tools")
    if isinstance(conv, list) and hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tools=tools,
            open_think=False,
            tokenize=False,
        )
        return str(rendered)
    return str(prompt_item)


class SftToolJsonGenerationProbeCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: Any,
        every_n_steps: int,
        max_new_tokens: int,
        prompts: Optional[List[Any]] = None,
        max_prompts_per_probe: int = 16,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.every_n_steps = max(int(every_n_steps), 0)
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.prompts = prompts or ['请输出合法 JSON：{"ok": true}']
        self.max_prompts_per_probe = max(1, int(max_prompts_per_probe))
        self._model_ref: torch.nn.Module | None = None

    @staticmethod
    def load_prompts_from_json(path: str | Path) -> List[Any]:
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("diag_sft_tool_json_prompts_json 应为 JSON 数组（元素可为字符串或 conversations 对象）")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._model_ref = model

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, float]] = None,
        model: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or self.every_n_steps <= 0:
            return
        step = int(state.global_step)
        if step <= 0 or step % self.every_n_steps != 0:
            return
        m = model if model is not None else self._model_ref
        if m is None:
            return
        m = _unwrap_model(m)
        device = next(m.parameters()).device
        tok = self.tokenizer
        m.eval()
        ok, total = 0, 0
        with torch.no_grad():
            for prompt in self.prompts[: self.max_prompts_per_probe]:
                prompt_text = _render_generation_prompt(tok, prompt)
                enc = tok(prompt_text, return_tensors="pt", add_special_tokens=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                out_ids = m.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    do_sample=False,
                )
                text = tok.decode(out_ids[0], skip_special_tokens=True)
                chunk = _first_balanced_brace_chunk(text)
                total += 1
                if chunk:
                    try:
                        json.loads(chunk)
                        ok += 1
                    except json.JSONDecodeError:
                        pass
        if total:
            rate = ok / total
            logs["sft_diag/tool_json_parse_rate"] = rate
            logger.info("sft_diag/tool_json_parse_rate step=%s ok=%s/%s", step, ok, total)
        self._swanlab_log(logs.get("sft_diag/tool_json_parse_rate"), step)

    @staticmethod
    def _swanlab_log(value: float | None, step: int) -> None:
        if value is None:
            return
        try:
            import swanlab
        except ImportError:
            return
        if swanlab.get_run() is None:
            return
        swanlab.log({"sft_diag/tool_json_parse_rate": float(value)}, step=step)
