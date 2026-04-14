from __future__ import annotations

import logging
import sys

from transformers import TrainingArguments
from trl import DPOConfig, DPOTrainer

from src.config.data_arguments import DpoDataArguments
from src.config.logging_config import setup_logging
from src.config.train_config import TrainConfig
from src.dataset.dpo_dataset import DPODataset
from src.ref_model import get_auto_tokenizer_local
from src.util.path_util import resolve_under_project
from src.model.model import MiniLmForCausalLM

logger = logging.getLogger(__name__)


def run_dpo(training_args: TrainingArguments, data_args: DpoDataArguments) -> None:

    if not data_args.pretrained_model_path:
        raise ValueError("pretrained_model_path 不能为空（DPO 需基于已训练策略模型）。")
    if not data_args.train_data_path:
        raise ValueError("train_data_path 不能为空。")

    tok_path = resolve_under_project(data_args.tokenizer_name_or_path)
    tokenizer = get_auto_tokenizer_local(tok_path, trust_remote_code=True)

    model_path = resolve_under_project(data_args.pretrained_model_path)
    model = MiniLmForCausalLM.from_pretrained(model_path)

    ref_model = None
    if data_args.ref_model_path:
        ref_path = resolve_under_project(data_args.ref_model_path)
        ref_model = MiniLmForCausalLM.from_pretrained(ref_path)

    train_data_path = resolve_under_project(data_args.train_data_path)
    train_dataset = DPODataset(train_data_path, tokenizer=tokenizer).as_hf_dataset()
    eval_dataset = None
    if training_args.do_eval and data_args.eval_data_path:
        eval_data_path = resolve_under_project(data_args.eval_data_path)
        eval_dataset = DPODataset(eval_data_path, tokenizer=tokenizer).as_hf_dataset()
    elif training_args.do_eval and not data_args.eval_data_path:
        logger.warning("do_eval=True 但未提供 eval_data_path，将跳过验证。")

    # TRL 1.x expects DPO-specific fields (beta/max_prompt_length/...) in DPOConfig(args),
    # not as standalone DPOTrainer kwargs.
    dpo_args = DPOConfig(
        **training_args.to_dict(),
        beta=float(data_args.dpo_beta),
        max_prompt_length=int(data_args.max_prompt_length),
        max_length=int(data_args.max_seq_length),
        truncation_mode=data_args.truncation_mode,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if training_args.should_save:
        logger.info("train loss: %s", train_result.training_loss)
        trainer.save_model()
        trainer.save_state()


def main() -> None:
    train_args = list(sys.argv[1:])
    logging.info("🚀 启动 MiniLM DPO 训练")
    training_args, data_args = TrainConfig.load_configs(
        "dpo",
        train_args=train_args,
    )
    run_dpo(training_args, data_args)


if __name__ == "__main__":
    setup_logging()
    main()

