"""
训练期诊断：验证批上的 next-token top-1、输出分布熵；定期无条件/前缀生成样例。
与分领域 eval（Trainer 对 eval_dataset dict）配合观察各领域 loss。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.monitor.common.rank_util import is_main_process

logger = logging.getLogger(__name__)


def _render_generation_prompt(tokenizer: Any, prompt_item: Any) -> str:
    """
    兼容两类格式：
    1) 纯字符串前缀；
    2) SFT 对话对象（与数据集 conversations 结构一致）：
       - 直接消息数组: [{role, content, ...}, ...]
       - 或对象: {"conversations":[...], "tools":[...]}
    """
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


def pick_probe_eval_dataset(eval_dataset: Any) -> Any:
    """多领域 eval 字典时，优先用键 eval，否则取第一个子集作为探针。"""
    if eval_dataset is None:
        return None
    if isinstance(eval_dataset, dict):
        if "eval" in eval_dataset:
            return eval_dataset["eval"]
        return next(iter(eval_dataset.values()))
    return eval_dataset


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _next_token_top1_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[float, float]:
    """logits (B,S,V), labels (B,S) 与 CLM 对齐：用 logits[:, :-1] 预测 labels[:, 1:]。"""
    if logits.dim() != 3 or labels.dim() != 2:
        return float("nan"), float("nan")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(ignore_index)
    if not mask.any():
        return float("nan"), float("nan")
    preds = shift_logits.argmax(dim=-1)
    correct = preds.eq(shift_labels) & mask
    top1 = correct.sum().float() / mask.sum().float()

    p = torch.nn.functional.softmax(shift_logits.float(), dim=-1)
    ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)
    mean_ent = (ent * mask.float()).sum() / mask.sum().float()
    return float(top1.item()), float(mean_ent.item())


class TrainingDiagnosticsCallback(TrainerCallback):
    """
    - every_n_steps：在 eval_dataset 上取 1～N 个 micro-batch（经同一 data_collator），记录 top-1、熵。
    - gen_every_n_steps：对 gen_prompts 做 greedy 生成，写入日志；若 SwanLab 已 init 则同步标量/文本。
    """

    def __init__(
        self,
        tokenizer: Any,
        data_collator: Any,
        eval_dataset: Any | None,
        ignore_index: int = -100,
        every_n_steps: int = 0,
        num_eval_batches: int = 1,
        gen_every_n_steps: int = 0,
        gen_max_new_tokens: int = 64,
        gen_prompts: Optional[List[Any]] = None,
        gen_temperature: float = 1.0,
        gen_do_sample: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.ignore_index = ignore_index
        self.every_n_steps = max(int(every_n_steps), 0)
        self.num_eval_batches = max(int(num_eval_batches), 1)
        self.gen_every_n_steps = max(int(gen_every_n_steps), 0)
        self.gen_max_new_tokens = max(int(gen_max_new_tokens), 1)
        self.gen_prompts = gen_prompts or [
            "以下是本文的主要内容：",
            "很久很久以前，在一个很远的地方，",
            "北京是中国的首都，它以",
        ]
        self.gen_temperature = float(gen_temperature)
        self.gen_do_sample = bool(gen_do_sample)
        self._model_ref: torch.nn.Module | None = None

    @staticmethod
    def load_prompts_from_json(path: str | Path) -> List[Any]:
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("diag_gen_prompts_json 应为 JSON 数组（元素可为字符串或 conversations 对象）")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        self._model_ref = model

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        model: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if not is_main_process():
            return
        if logs is None:
            return
        step = int(state.global_step)
        m = model if model is not None else self._model_ref
        if m is None:
            return

        if self.every_n_steps and step > 0 and step % self.every_n_steps == 0 and self.eval_dataset is not None:
            self._run_eval_batch_metrics(m, logs, step)

        if self.gen_every_n_steps and step > 0 and step % self.gen_every_n_steps == 0:
            self._run_generation(m, step)

    def _run_eval_batch_metrics(self, model: torch.nn.Module, logs: Dict[str, float], step: int) -> None:
        from torch.utils.data import DataLoader

        ds = self.eval_dataset
        if len(ds) == 0:
            return
        collator = self.data_collator
        batch_size = max(1, min(4, len(ds)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
        device = next(model.parameters()).device
        m = _unwrap_model(model)
        was_training = m.training
        m.eval()
        top1_list: list[float] = []
        ent_list: list[float] = []
        try:
            with torch.no_grad():
                for i, features in enumerate(dl):
                    if i >= self.num_eval_batches:
                        break
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
                    out = m(**batch)
                    logits = out.logits
                    labels = batch["labels"]
                    t1, h = _next_token_top1_and_entropy(logits, labels, self.ignore_index)
                    if not (t1 != t1):  # not NaN
                        top1_list.append(t1)
                    if not (h != h):
                        ent_list.append(h)
        finally:
            if was_training:
                m.train()
        if top1_list:
            logs["diag/eval_top1"] = sum(top1_list) / len(top1_list)
        if ent_list:
            logs["diag/eval_entropy"] = sum(ent_list) / len(ent_list)
        self._swanlab_log({"diag/eval_top1": logs.get("diag/eval_top1"), "diag/eval_entropy": logs.get("diag/eval_entropy")}, step)

    def _run_generation(self, model: torch.nn.Module, step: int) -> None:
        m = _unwrap_model(model)
        device = next(model.parameters()).device
        was_training = m.training
        m.eval()
        tok = self.tokenizer
        max_new = self.gen_max_new_tokens
        payload: Dict[str, Any] = {}
        try:
            with torch.no_grad():
                for pi, prompt in enumerate(self.gen_prompts[:8]):
                    prompt_text = _render_generation_prompt(tok, prompt)
                    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=True)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    gen_kw: Dict[str, Any] = dict(
                        max_new_tokens=max_new,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                    if self.gen_do_sample:
                        gen_kw["do_sample"] = True
                        gen_kw["temperature"] = self.gen_temperature
                    else:
                        gen_kw["do_sample"] = False
                    out_ids = m.generate(**enc, **gen_kw)
                    text = tok.decode(out_ids[0], skip_special_tokens=True)
                    key = f"diag/gen/prompt_{pi}"
                    payload[key] = text
                    logger.info("[%s] step=%s %s", key, step, text[:500].replace("\n", "\\n"))
        finally:
            if was_training:
                m.train()
        self._swanlab_log_text(payload, step)

    @staticmethod
    def _swanlab_log(scalars: Dict[str, Any], step: int) -> None:
        try:
            import swanlab
        except ImportError:
            return
        if swanlab.get_run() is None:
            return
        clean = {k: v for k, v in scalars.items() if v is not None and isinstance(v, (int, float))}
        if clean:
            swanlab.log(clean, step=step)

    @staticmethod
    def _swanlab_log_text(payload: Dict[str, Any], step: int) -> None:
        try:
            import swanlab
        except ImportError:
            return
        if swanlab.get_run() is None:
            return
        for k, v in payload.items():
            if isinstance(v, str):
                try:
                    swanlab.log({k: swanlab.Text(v)}, step=step)
                except Exception:
                    swanlab.log({k: v[:2000]}, step=step)
