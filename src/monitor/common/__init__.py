"""预训练与 SFT 共用的训练回调与诊断工具。"""

from src.monitor.common.diagnostics_metrics import (
    attention_head_entropy,
    summarize_attentions,
    summarize_hidden_states,
)
from src.monitor.common.loss_grad_callbacks import GradNormPostClipCallback, LossNormalizeCallback
from src.monitor.common.training_diagnostics_callback import (
    TrainingDiagnosticsCallback,
    pick_probe_eval_dataset,
)

__all__ = [
    "GradNormPostClipCallback",
    "LossNormalizeCallback",
    "TrainingDiagnosticsCallback",
    "attention_head_entropy",
    "pick_probe_eval_dataset",
    "summarize_attentions",
    "summarize_hidden_states",
]
