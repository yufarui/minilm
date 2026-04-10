#!/usr/bin/env bash
# 预训练语料预处理（kind: pretrain 的任务 YAML）。
# 请在项目根目录执行，或任意目录下传入脚本绝对路径。
#
# 无参数时：使用环境变量 PREPROCESS_CONFIG（默认 config/preprocess/pipeline.job.yaml）。
# 有参数时：原样传给 ``python -m src.preprocess.run_preprocess``（可完全自行指定 --config 等）。
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

if [[ $# -gt 0 ]]; then
  exec python -m src.preprocess.run_preprocess "$@"
else
  exec python -m src.preprocess.run_preprocess --config "${DEFAULT_CONFIG}"
fi
