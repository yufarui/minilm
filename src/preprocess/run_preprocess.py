"""
预训练 / SFT 共用预处理 CLI（策略由任务 YAML 的 ``kind`` 选择）。

主入口为 **任务 YAML 文件**（含 input/output、kind、pipeline 或 sft 嵌套配置；标准 JSON 亦可）。

示例::

    python -m src.preprocess.run_preprocess --config config/preprocess/pipeline.job.yaml
    python -m src.preprocess.run_preprocess --config config/preprocess/sft.pipeline.job.yaml

可选依赖：langdetect、datasketch、fasttext（SFT ``lang_backend: fasttext``）、bertopic（见 pyproject optional-dependencies）。
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.preprocess.job_config import PreprocessJobFile
from src.preprocess.split_dataset import (
    iter_jsonl,
    split_pretrain_train_val,
    split_sft_train_and_eval_sets,
    write_jsonl,
)
from src.preprocess.stats_plots import save_preprocess_charts
from src.preprocess.strategies.pretrain import PretrainPreprocessStrategy
from src.preprocess.strategies.sft import SftPreprocessStrategy


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniLM 预训练 / SFT 语料预处理（策略模式）")
    p.add_argument(
        "--config",
        required=True,
        help="任务 YAML：含 input、output、kind（pretrain|sft）、stats_path、pipeline 或 sft",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    job = PreprocessJobFile.load(args.config)

    if job.kind == "sft":
        strategy = SftPreprocessStrategy(job.sft_pipeline_for_run())
        train_output = job.resolved_output()
        stats = strategy.run(job.resolved_input(), train_output)
        if job.sft_split.enabled:
            rows = list(iter_jsonl(train_output))
            train_rows, tool_rows, multi_rows = split_sft_train_and_eval_sets(
                rows=rows,
                tool_val_size=job.sft_split.tool_call_val_size,
                multi_turn_val_size=job.sft_split.multi_turn_val_size,
                seed=job.sft_split.seed,
                conversations_field=job.sft.conversations_field,
            )
            tool_out = job.resolved_sft_tool_val_output()
            multi_out = job.resolved_sft_multi_turn_val_output()
            write_jsonl(train_output, train_rows)
            write_jsonl(tool_out, tool_rows)
            write_jsonl(multi_out, multi_rows)
            logging.info(
                "SFT split done: train=%s tool_val=%s (%s) multi_turn_val=%s (%s)",
                len(train_rows),
                len(tool_rows),
                tool_out,
                len(multi_rows),
                multi_out,
            )
        logging.info("SFT stats: %s", stats)
        plot_dir = job.resolved_plots_dir()
        if plot_dir is not None:
            logging.info("kind=sft：未生成预训练风格图表；详见 stats JSON 中 diagnostics 字段")
        return 0

    pipe_cfg = job.pipeline_for_run()
    strategy = PretrainPreprocessStrategy(pipe_cfg)
    train_output = job.resolved_output()
    stats = strategy.run(job.resolved_input(), train_output)
    if job.pretrain_split.enabled:
        rows = list(iter_jsonl(train_output))
        train_rows, val_rows = split_pretrain_train_val(
            rows=rows,
            val_size=job.pretrain_split.val_size,
            seed=job.pretrain_split.seed,
        )
        val_out = job.resolved_pretrain_val_output()
        write_jsonl(train_output, train_rows)
        write_jsonl(val_out, val_rows)
        logging.info(
            "Pretrain split done: train=%s val=%s (%s)",
            len(train_rows),
            len(val_rows),
            val_out,
        )
    logging.info("pretrain stats: %s", stats)

    plot_dir = job.resolved_plots_dir()
    if plot_dir is not None:
        paths = save_preprocess_charts(stats, pipe_cfg, plot_dir)
        logging.info("统计图已写入 %s ：%s", plot_dir, [p.name for p in paths])

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
