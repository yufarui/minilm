from __future__ import annotations

import json
import logging
import sys

from transformers import Trainer, TrainerCallback, TrainingArguments

from src.config.data_arguments import SftDataArguments
from src.config.logging_config import setup_logging
from src.config.model_config import MiniLMConfig
from src.config.train_config import TrainConfig
from src.dataset.sft_dataset import SFTDataset
from src.model.model import MiniLmForCausalLM
from src.monitor.sft.build_callbacks import build_sft_trainer_callbacks
from src.ref_model import get_auto_tokenizer_local
from src.util.data_collator import TrainDataCollator
from src.util.path_util import resolve_under_project

logger = logging.getLogger(__name__)


def run_sft(training_args: TrainingArguments, data_args: SftDataArguments) -> None:
    tok_path = resolve_under_project(data_args.tokenizer_name_or_path)
    tokenizer = get_auto_tokenizer_local(tok_path, trust_remote_code=True)

    if data_args.pretrained_model_path:
        pretrained_path = resolve_under_project(data_args.pretrained_model_path)
        model = MiniLmForCausalLM.from_pretrained(pretrained_path)
    else:
        cfg_path = resolve_under_project(data_args.model_config_file)
        model_config = MiniLMConfig.from_pretrained(cfg_path)
        model = MiniLmForCausalLM(model_config)

    train_data_path = resolve_under_project(data_args.train_data_path)
    train_dataset = SFTDataset(
        train_data_path,
        tokenizer,
        pack_bin_size=data_args.max_seq_length,
    )
    eval_dataset = None
    if training_args.do_eval:
        eval_map: dict[str, SFTDataset] = {}
        if data_args.eval_data_path:
            eval_data_path = resolve_under_project(data_args.eval_data_path)
            eval_map["eval"] = SFTDataset(
                eval_data_path,
                tokenizer,
                pack_bin_size=data_args.max_seq_length,
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
                eval_map[str(domain_name)] = SFTDataset(
                    eval_data_file,
                    tokenizer,
                    pack_bin_size=data_args.max_seq_length,
                )
        if not eval_map:
            logger.warning("do_eval=True 但未设置 eval_data_path 或 eval_domains_json，eval_dataset=None")
        elif len(eval_map) == 1:
            eval_dataset = next(iter(eval_map.values()))
        else:
            eval_dataset = eval_map

    data_collator = TrainDataCollator(tokenizer)
    extra_callbacks: list[TrainerCallback] = build_sft_trainer_callbacks(
        training_args,
        data_args,
        tokenizer,
        data_collator,
        eval_dataset,
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=extra_callbacks or None,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if training_args.should_save:
        logger.info("train loss: %s", train_result.training_loss)
        trainer.save_model()
        trainer.save_state()


def main() -> None:
    train_args = list(sys.argv[1:])
    logging.info("🚀 启动 MiniLM 全参数 SFT 训练")

    training_args, data_args = TrainConfig.load_configs(
        "sft",
        train_args=train_args,
    )

    run_sft(training_args, data_args)


if __name__ == "__main__":
    setup_logging()
    main()
