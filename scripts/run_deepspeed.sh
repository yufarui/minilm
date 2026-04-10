#!/usr/bin/env bash
# DeepSpeed 启动训练。请在项目根目录执行。
# 需安装: pip install -e ".[deepspeed]" 或 pip install deepspeed
#
# 环境变量:
#   STAGE              同 run_single_gpu.sh，默认 pretrain
#   NUM_GPUS           DeepSpeed 使用的 GPU 数量，默认 1
#   DEEPSPEED_CONFIG   DeepSpeed JSON，默认 config/pretrain/deepspeed_zero2.json
#
# 默认将 --deepspeed 指向 DEEPSPEED_CONFIG；训练/数据 JSON 仍由各阶段默认路径或 CLI 覆盖。
#
# 示例:
#   cd /path/to/minilm
#   STAGE=pretrain NUM_GPUS=2 bash scripts/run_deepspeed.sh
#   STAGE=sft NUM_GPUS=2 bash scripts/run_deepspeed.sh
#   STAGE=dpo DEEPSPEED_CONFIG=config/pretrain/deepspeed_zero2.json bash scripts/run_deepspeed.sh --max_steps 500
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_train_resolve.sh
source "${SCRIPT_DIR}/_train_resolve.sh"

cd "${ROOT}"
STAGE="${STAGE:-pretrain}"
MOD="$(resolve_stage_to_module "${STAGE}")" || exit 1

NUM_GPUS="${NUM_GPUS:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/pretrain/deepspeed_zero2.json}"

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "未找到 deepspeed 命令，请先安装: pip install -e '.[deepspeed]'" >&2
  exit 1
fi

exec deepspeed \
  --num_gpus="${NUM_GPUS}" \
  --module "${MOD}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  "$@"
