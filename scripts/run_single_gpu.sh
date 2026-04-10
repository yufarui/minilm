#!/usr/bin/env bash
# 单 GPU 启动训练（在项目根目录执行，或任意目录下传入脚本绝对路径）。
#
# 场景由环境变量 STAGE 决定，对应入口模块：
#   pretrain -> python -m src.trainer.train_pretrain
#   sft      -> python -m src.trainer.train_full_sft
#   dpo      -> python -m src.trainer.train_dpo
#
# 示例:
#   cd /path/to/minilm
#   STAGE=pretrain bash scripts/run_single_gpu.sh
#   STAGE=sft bash scripts/run_single_gpu.sh --learning_rate 1e-5
#   STAGE=dpo bash scripts/run_single_gpu.sh --train_args_file config/dpo/train_args.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_train_resolve.sh
source "${SCRIPT_DIR}/_train_resolve.sh"

cd "${ROOT}"
STAGE="${STAGE:-pretrain}"
MOD="$(resolve_stage_to_module "${STAGE}")" || exit 1

exec python -m "${MOD}" "$@"
