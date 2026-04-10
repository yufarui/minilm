from __future__ import annotations

import torch
from transformers import TrainerCallback


class LossNormalizeCallback(TrainerCallback):
    """将日志中的 loss 归一化到每个 micro-step。"""

    def __init__(self, grad_accum_steps: int):
        self.grad_accum_steps = max(int(grad_accum_steps), 1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        try:
            raw = float(logs["loss"])
        except (TypeError, ValueError):
            return
        logs["loss_raw"] = raw
        logs["loss"] = raw / self.grad_accum_steps


class GradNormPostClipCallback(TrainerCallback):
    """记录优化器 step 前（已裁剪后）的梯度范数，便于与日志中的 pre-clip grad_norm 对比。"""

    def __init__(self):
        self.last_post_clip_grad_norm: float | None = None

    @staticmethod
    def _total_grad_norm(model: torch.nn.Module) -> float | None:
        total = 0.0
        found = False
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            param_norm = g.norm(2).item()
            total += param_norm * param_norm
            found = True
        if not found:
            return None
        return total ** 0.5

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        self.last_post_clip_grad_norm = self._total_grad_norm(model)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.last_post_clip_grad_norm is None:
            return
        logs["grad_norm_post_clip"] = self.last_post_clip_grad_norm
