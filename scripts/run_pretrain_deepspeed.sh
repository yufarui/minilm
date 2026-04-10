#!/usr/bin/env bash
# DeepSpeed 单机多卡。train_type 由入口 Python 模块决定；ZeRO 通过 --deepspeed 追加。
#
# 环境变量:
#   NUM_GPUS           默认 2
#   DEEPSPEED_CONFIG   默认 config/pretrain/deepspeed_zero2.json
#   TRAIN_TARGET, TRAIN_SCRIPT  同 train_stage.inc.sh
#
# 执行示例（建议在 `/minilm` 下执行）：
#   NUM_GPUS=8 DEEPSPEED_CONFIG=config/pretrain/deepspeed_zero3.json \
#     SWANLAB_PROJECT=xxx SWANLAB_MODE=xxx SWANLAB_API_KEY=xxx \
#     ./scripts/run_pretrain_deepspeed.sh
set -euo pipefail
#
# 建议在项目根（例如你的 Linux 路径 `/minilm`）下执行。
# 为了避免“在别的目录执行脚本”导致路径解析出错，这里优先用当前目录推导 ROOT。
if [[ -f "pyproject.toml" ]]; then
  ROOT="$(pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"
# shellcheck source=scripts/include/train_stage.inc.sh
source "${ROOT}/scripts/include/train_stage.inc.sh"
source "${ROOT}/scripts/include/swanlab_env.sh"
NUM_GPUS="${NUM_GPUS:-2}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/pretrain/deepspeed_zero2.json}"
exec deepspeed --num_gpus="${NUM_GPUS}" "${TRAIN_SCRIPT_REL}" \
  --deepspeed "${DEEPSPEED_CONFIG}" "$@"
