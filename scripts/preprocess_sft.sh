#!/usr/bin/env bash
# SFT 对话 JSONL 预处理（kind: sft 的任务 YAML）。
# 请在项目根目录执行，或任意目录下传入脚本绝对路径。
#
# 无参数时：使用环境变量 PREPROCESS_CONFIG（默认 config/preprocess/sft.pipeline.job.yaml）。
# 有参数时：原样传给 ``python -m src.preprocess.run_preprocess``。
#
# 示例:
#   cd /path/to/minilm
#   bash scripts/preprocess_sft.sh
#   PREPROCESS_CONFIG=config/preprocess/sft.pipeline.job.yaml bash scripts/preprocess_sft.sh
#   bash scripts/preprocess_sft.sh --config config/preprocess/sft.pipeline.job.yaml
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DEFAULT_CONFIG="${PREPROCESS_CONFIG:-config/preprocess/sft.pipeline.job.yaml}"

if [[ $# -gt 0 ]]; then
  exec python -m src.preprocess.run_preprocess "$@"
else
  exec python -m src.preprocess.run_preprocess --config "${DEFAULT_CONFIG}"
fi
