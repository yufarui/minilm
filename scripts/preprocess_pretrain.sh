#!/usr/bin/env bash
# 预训练语料预处理（kind: pretrain 的任务 YAML）+ 默认 tokenized parquet 构建。
# 请在项目根目录执行，或任意目录下传入脚本绝对路径。
#
# 无参数时：使用环境变量 PREPROCESS_CONFIG（默认 config/preprocess/pipeline.job.yaml），
# 清洗切分后自动把训练集写成 config/pretrain/data_config.json 期望的
# data/pretrain/pretrain_train_arrow（scripts/to_arrow.py）。
# 有参数时：原样传给 ``python -m src.preprocess.run_preprocess``（不自动 to_arrow）。
#
# 示例:
#   cd /path/to/minilm
#   bash scripts/preprocess_pretrain.sh
#   PREPROCESS_CONFIG=config/preprocess/pipeline.job.yaml bash scripts/preprocess_pretrain.sh
#   bash scripts/preprocess_pretrain.sh --config config/preprocess/pipeline.job.yaml
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DEFAULT_CONFIG="${PREPROCESS_CONFIG:-config/preprocess/pipeline.job.yaml}"
DEFAULT_TRAIN_JSONL="data/pretrain/pretrain_train.jsonl"
DEFAULT_TRAIN_ARROW="data/pretrain/pretrain_train_arrow"
DEFAULT_TOKENIZER="tokenizer/minilm"

if [[ $# -gt 0 ]]; then
  exec python -m src.preprocess.run_preprocess "$@"
fi

python -m src.preprocess.run_preprocess --config "${DEFAULT_CONFIG}"

python scripts/to_arrow.py \
  --jsonl_path "${DEFAULT_TRAIN_JSONL}" \
  --output_path "${DEFAULT_TRAIN_ARROW}" \
  --tokenizer_name_or_path "${DEFAULT_TOKENIZER}"
