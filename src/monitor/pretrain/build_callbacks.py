"""预训练 Trainer 回调列表（MoE SwanLab、loss 归一化、验证集 top-1/熵、生成探针）。"""

from __future__ import annotations

import logging
from typing import Any

from transformers import TrainerCallback, TrainingArguments

from src.config.data_arguments import PretrainDataArguments
from src.monitor.common.loss_grad_callbacks import GradNormPostClipCallback, LossNormalizeCallback
from src.monitor.common.training_diagnostics_callback import (
    TrainingDiagnosticsCallback,
    pick_probe_eval_dataset,
)
from src.monitor.pretrain.swanlab_moe_callback import MiniLMSwanlabDiagCallback
from src.util.path_util import resolve_under_project

logger = logging.getLogger(__name__)


def build_pretrain_trainer_callbacks(
    training_args: TrainingArguments,
    data_args: PretrainDataArguments,
    tokenizer: Any,
    data_collator: Any,
    eval_dataset: Any,
) -> list[TrainerCallback]:
    extra_callbacks: list[TrainerCallback] = [
        MiniLMSwanlabDiagCallback(),
        LossNormalizeCallback(training_args.gradient_accumulation_steps),
        GradNormPostClipCallback(),
    ]

    if data_args.diag_every_n_steps > 0 and pick_probe_eval_dataset(eval_dataset) is None:
        logger.warning("diag_every_n_steps>0 但未配置可用验证集，top-1/熵诊断将跳过")

    gen_prompts = None
    if data_args.diag_gen_prompts_json:
        gen_prompts = TrainingDiagnosticsCallback.load_prompts_from_json(
            resolve_under_project(data_args.diag_gen_prompts_json)
        )

    if data_args.diag_every_n_steps > 0 or data_args.diag_gen_every_n_steps > 0:
        extra_callbacks.append(
            TrainingDiagnosticsCallback(
                tokenizer=tokenizer,
                data_collator=data_collator,
                eval_dataset=pick_probe_eval_dataset(eval_dataset),
                every_n_steps=data_args.diag_every_n_steps,
                num_eval_batches=data_args.diag_num_eval_batches,
                gen_every_n_steps=data_args.diag_gen_every_n_steps,
                gen_max_new_tokens=data_args.diag_gen_max_new_tokens,
                gen_prompts=gen_prompts,
                gen_temperature=data_args.diag_gen_temperature,
                gen_do_sample=data_args.diag_gen_do_sample,
            )
        )

    return extra_callbacks
