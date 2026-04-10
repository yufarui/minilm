"""
训练与数据配置: JSON + CLI override（CLI 仅覆盖命令行显式出现的字段）。
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Tuple, Type, TypeVar

from transformers import HfArgumentParser, TrainingArguments

from src.util.path_util import resolve_under_project

from .data_arguments import DpoDataArguments, PretrainDataArguments, SftDataArguments, TrainDataArguments

T = TypeVar("T")
D = TypeVar("D", bound=TrainDataArguments)

logger = logging.getLogger(__name__)


@dataclass
class TrainScriptArguments:
    """入口：可选覆盖训练/数据 JSON 路径（``train_type`` 由调用方传入 ``load_configs``，不从 CLI 解析）。"""

    train_args_file: Optional[str] = field(
        default=None,
        metadata={"help": "覆盖默认训练参数 JSON"},
    )
    data_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "覆盖默认数据配置 JSON"},
    )


class TrainConfig:
    """多阶段训练：在 ``config_files`` 中为每个阶段登记 ``train_args`` / ``data_config`` JSON。

    ``train_type`` 仅由 ``load_configs(train_type, ...)`` 的位置参数传入；不由 CLI 解析。
    """

    config_files = {
        "pre_train": {
            "train_args": "config/pretrain/train_args.json",
            "data_config": "config/pretrain/data_config.json",
            "data_args_cls": PretrainDataArguments,
        },
        "sft": {
            "train_args": "config/sft/train_args.json",
            "data_config": "config/sft/data_config.json",
            "data_args_cls": SftDataArguments,
        },
        "dpo": {
            "train_args": "config/dpo/train_args.json",
            "data_config": "config/dpo/data_config.json",
            "data_args_cls": DpoDataArguments,
        },
    }

    @classmethod
    def load_configs(
            cls,
            train_type: str,
            train_args: Optional[list[str]] = None,
    ) -> Tuple[TrainingArguments, TrainDataArguments]:
        """加载训练/数据 JSON；``train_type`` 由调用方代码指定。"""

        if train_args is None:
            train_args = list(sys.argv[1:])

        script_parser = HfArgumentParser(TrainScriptArguments)
        script_args, rest = script_parser.parse_args_into_dataclasses(
            args=train_args,
            look_for_args_file=False,
            return_remaining_strings=True,
        )

        effective = train_type
        if not effective or effective not in cls.config_files:
            raise KeyError(
                f"未知 train_type={effective!r}，可选: {list(cls.config_files)}。"
                "新增阶段请在 TrainConfig.config_files 注册。"
            )

        cfg_paths = cls.config_files[effective]
        data_cls: Type[D] = cfg_paths["data_args_cls"]

        train_path = resolve_under_project(script_args.train_args_file or cfg_paths["train_args"])
        data_path = resolve_under_project(script_args.data_config_file or cfg_paths["data_config"])
        logger.info(
            "训练阶段 train_type=%s train_args=%s data_config=%s",
            effective,
            train_path,
            data_path,
        )

        train_parser = HfArgumentParser(TrainingArguments)
        with open(train_path, encoding="utf-8") as f:
            train_dict = json.load(f)
        ds_cfg = train_dict.get("deepspeed")
        if isinstance(ds_cfg, str):
            train_dict["deepspeed"] = str(resolve_under_project(ds_cfg))
        (training_args,) = train_parser.parse_dict(train_dict)

        data_parser = HfArgumentParser(data_cls)
        (data_args,) = data_parser.parse_json_file(json_file=data_path)

        if not rest:
            return training_args, data_args

        cli_parser = HfArgumentParser((TrainingArguments, data_cls))
        train_cli, data_cli = cli_parser.parse_args_into_dataclasses(
            args=list(rest),
            look_for_args_file=False,
        )

        explicit = cls._explicit_argparse_dests(rest, cli_parser)

        training_args = cls._merge_json_cli_dataclass(
            training_args, train_cli, TrainingArguments, explicit
        )
        data_args = cls._merge_json_cli_dataclass(data_args, data_cli, data_cls, explicit)

        return training_args, data_args

    @staticmethod
    def _explicit_argparse_dests(argv: list[str], parser: HfArgumentParser) -> set[str]:
        """根据 argv 中出现的长选项，解析出对应的 argparse ``dest``（用于判断用户是否显式传入）。"""
        token_prefixes: set[str] = set()
        for t in argv:
            if not t.startswith("--"):
                continue
            token_prefixes.add(t.split("=", 1)[0])

        dests: set[str] = set()
        for action in parser._actions:
            opts = getattr(action, "option_strings", None) or ()
            for opt in opts:
                if opt in token_prefixes:
                    dest = getattr(action, "dest", None)
                    if dest and dest != "help":
                        dests.add(dest)
        return dests

    @staticmethod
    def _merge_json_cli_dataclass(
            base: T,
            cli: T,
            dtype: Type[T],
            explicit_dests: set[str],
    ) -> T:
        """仅用 CLI 中显式出现的字段覆盖 JSON 解析结果。"""
        base_d = asdict(base)
        cli_d = asdict(cli)
        names = {f.name for f in fields(dtype)}
        for k in explicit_dests:
            if k in names and k in cli_d:
                base_d[k] = cli_d[k]
        if dtype is TrainingArguments and isinstance(base_d.get("deepspeed"), str):
            base_d["deepspeed"] = str(resolve_under_project(base_d["deepspeed"]))
        return type(base)(**base_d)
