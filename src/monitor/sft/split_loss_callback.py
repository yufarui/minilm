"""将自定义分项 loss（若存在）镜像到统一前缀，便于 SwanLab / W&B 分组。"""

from __future__ import annotations

from transformers import TrainerCallback

from src.monitor.common.rank_util import is_main_process


class SftSplitLossMirrorCallback(TrainerCallback):
    """
    当训练循环在 ``logs`` 中提供 ``loss_text``、``loss_tool``、``loss_think`` 等字段时，
    额外写入 ``sft/loss_text`` 等。当前标准 HF Trainer 仅聚合 ``loss``；分项需在自定义
    ``Trainer.compute_loss`` 或模型 forward 返回值中自行注入日志键后才会出现。
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process():
            return
        if not logs:
            return
        for k in ("loss_text", "loss_tool", "loss_think"):
            if k in logs and logs[k] is not None:
                try:
                    logs[f"sft/{k}"] = float(logs[k])
                except (TypeError, ValueError):
                    pass
