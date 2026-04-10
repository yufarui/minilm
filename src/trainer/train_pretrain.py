from __future__ import annotations

import json
import logging
import sys

from transformers import TrainerCallback, TrainingArguments, Trainer

from src.config.data_arguments import PretrainDataArguments
from src.config.logging_config import setup_logging
from src.config.model_config import MiniLMConfig
from src.config.train_config import TrainConfig
from src.dataset.pre_train_dataset import PreTrainDataset
from src.model.model import MiniLmForCausalLM
from src.monitor.pretrain.build_callbacks import build_pretrain_trainer_callbacks
from src.ref_model import get_auto_tokenizer_local
from src.util.data_collator import TrainDataCollator
from src.util.path_util import resolve_under_project

logger = logging.getLogger(__name__)


def run_pretrain(training_args: TrainingArguments, data_args: PretrainDataArguments) -> None:
    tok_path = resolve_under_project(data_args.tokenizer_name_or_path)
    tokenizer = get_auto_tokenizer_local(tok_path, trust_remote_code=True)

    cfg_path = resolve_under_project(data_args.model_config_file)
    model_config = MiniLMConfig.from_pretrained(cfg_path)
    model = MiniLmForCausalLM(model_config)

    pack_bin_size = data_args.pack_bin_size or data_args.max_seq_length
    train_data_path = resolve_under_project(data_args.train_data_path)
    train_dataset = PreTrainDataset(
        train_data_path,
        tokenizer,
        pack_bin_size=pack_bin_size,
        pack_bin_schedule=data_args.pack_bin_schedule,
    )
    eval_dataset = None
    if training_args.do_eval:
        eval_map: dict[str, PreTrainDataset] = {}
        if data_args.eval_data_path:
            eval_data_path = resolve_under_project(data_args.eval_data_path)
            eval_map["eval"] = PreTrainDataset(
                eval_data_path,
                tokenizer,
                pack_bin_size=pack_bin_size,
                pack_bin_schedule=None,
            )
        if data_args.eval_domains_json:
            domains_manifest_path = resolve_under_project(data_args.eval_domains_json)
            with open(domains_manifest_path, encoding="utf-8") as manifest_fp:
                domain_to_path = json.load(manifest_fp)
            if not isinstance(domain_to_path, dict):
                raise ValueError(
                    "eval_domains_json 指向的 JSON 顶层须为对象：键=领域名，值=该领域验证数据文件路径（JSONL）"
                )
            for domain_name, path in domain_to_path.items():
                eval_data_file = resolve_under_project(str(path))
                eval_map[str(domain_name)] = PreTrainDataset(
                    eval_data_file,
                    tokenizer,
                    pack_bin_size=pack_bin_size,
                    pack_bin_schedule=None,
                )
        if not eval_map:
            logger.warning("do_eval=True 但未设置 eval_data_path 或 eval_domains_json，eval_dataset=None")
        elif len(eval_map) == 1:
            eval_dataset = next(iter(eval_map.values()))
        else:
            eval_dataset = eval_map

    data_collator = TrainDataCollator(tokenizer)

    extra_callbacks: list[TrainerCallback] = build_pretrain_trainer_callbacks(
        training_args,
        data_args,
        tokenizer,
        data_collator,
        eval_dataset,
    )

    if training_args.resume_from_checkpoint:
        logger.info(
            "断点续训：HuggingFace Trainer 会按 checkpoint 中的 global_step 跳过已消费的 micro-batch，"
            "并恢复随机数状态；须与保存 checkpoint 时保持一致：train_data_path 内容、"
            "pack_bin_schedule、tokenizer、本仓库 packing 代码及 "
            "per_device_train_batch_size / gradient_accumulation_steps / 分布式 world_size，"
            "否则样本序列与优化器步进可能错位。"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=extra_callbacks or None,
    )

    if training_args.resume_from_checkpoint:
        logger.info(
            "断点续训：Trainer 将跳过已在 checkpoint 中完成的 micro-batch，并恢复 RNG；"
            "请保持 data_config（语料路径、pack_bin_schedule）与 "
            "train_args（batch、GAS、dataloader_drop_last、seed/data_seed）与保存 checkpoint 时一致。"
        )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if training_args.should_save:
        logger.info("train loss: %s", train_result.training_loss)
        trainer.save_model()
        trainer.save_state()


def main() -> None:
    train_args = list(sys.argv[1:])

    logging.info("🚀 启动 MiniLM 预训练")

    training_args, data_args = TrainConfig.load_configs(
        "pre_train",
        train_args=train_args,
    )

    run_pretrain(training_args, data_args)


if __name__ == "__main__":
    setup_logging()
    main()
