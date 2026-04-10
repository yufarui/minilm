
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import swanlab
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.model.moe import Moe


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _inner_minilm_model(model: Optional[torch.nn.Module]) -> Optional[torch.nn.Module]:
    """MiniLmForCausalLM -> .model；已是 MiniLMModel 则原样返回。"""
    if model is None:
        return None
    m = _unwrap_model(model)
    return getattr(m, "model", m)


class MiniLMSwanlabDiagCallback(TrainerCallback):
    """
    - 在每次 `on_log` 时写入 MoE：各层 aux_loss、门控统计的各专家 token 占比（来自最近一次 forward 的 hook）。
    - 每隔 `log_attn_every` 步，若提供 `attn_probe`，对 MiniLMModel 做一次 `output_attentions=True` 并记录熵。
    - 每隔 `log_hidden_every` 步，使用同一 `attn_probe` 输入记录各层 hidden 统计（无需注意力时可不传 output_attentions）。

    attn_probe 示例：{"input_ids": LongTensor[1, L], "attention_mask": optional}
    """

    def __init__(
            self,
            log_gate_expert_scalars: bool = True,
            max_gate_experts_as_scalars: int = 32,
    ) -> None:
        super().__init__()
        self.log_gate_expert_scalars = log_gate_expert_scalars
        self.max_gate_experts_as_scalars = max_gate_experts_as_scalars
        self._handles: List[Any] = []
        self._last_gate_frac: Dict[int, torch.Tensor] = {}

    def _gate_hook(self, layer_idx: int) -> Callable[..., None]:
        def _fn(module: torch.nn.Module, _args: Any, output: Any) -> None:
            topk_idx, _w, _aux = output
            n = module.n_routed_experts
            flat = topk_idx.reshape(-1).long()
            h = torch.bincount(flat, minlength=n).float()
            h = h / h.sum().clamp_min(1e-12)
            self._last_gate_frac[layer_idx] = h.detach().cpu()

        return _fn

    def _register_moe_hooks(self, model: Optional[torch.nn.Module]) -> None:
        self._remove_moe_hooks()
        inner = _inner_minilm_model(model)
        if inner is None or not hasattr(inner, "layers"):
            return
        for i, layer in enumerate(inner.layers):
            if isinstance(getattr(layer, "mlp", None), Moe):
                h = layer.mlp.gate.register_forward_hook(self._gate_hook(i))
                self._handles.append(h)

    def _remove_moe_hooks(self) -> None:

        for h in self._handles:
            h.remove()
        self._handles.clear()

    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model: Optional[torch.nn.Module] = None,
            **kwargs: Any,
    ) -> None:
        self._register_moe_hooks(model)

    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
    ) -> None:
        self._remove_moe_hooks()

    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: Optional[Dict[str, float]] = None,
            model: Optional[torch.nn.Module] = None,
            **kwargs: Any,
    ) -> None:

        if swanlab.get_run() is None or logs is None:
            return

        step = state.global_step
        inner = _inner_minilm_model(model)
        if inner is None:
            return

        payload: Dict[str, Any] = {}

        total_aux: Optional[torch.Tensor] = None
        for i, layer in enumerate(inner.layers):
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, Moe):
                v = mlp.aux_loss
                if isinstance(v, torch.Tensor):
                    total_aux = v if total_aux is None else total_aux + v
                    payload[f"moe/aux_loss_layer_{i}"] = float(v.detach().float().item())
        if total_aux is not None:
            payload["moe/aux_loss_total"] = float(total_aux.detach().float().item())

        for li, frac in self._last_gate_frac.items():
            ne = int(frac.numel())
            if self.log_gate_expert_scalars and ne <= self.max_gate_experts_as_scalars:
                for e in range(ne):
                    payload[f"moe/gate_frac/layer_{li}/expert_{e}"] = float(frac[e].item())
            else:
                p = frac.clamp_min(1e-12)
                payload[f"moe/gate_frac/layer_{li}/mean"] = float(frac.mean().item())
                payload[f"moe/gate_frac/layer_{li}/std"] = float(frac.std().item())
                payload[f"moe/gate_frac/layer_{li}/entropy"] = float(-(p * p.log()).sum().item())

        if payload:
            swanlab.log(payload, step=step)
