#!/usr/bin/env bash
# 单机单卡。配置阶段由入口模块内 TrainConfig.load_configs(train_type, …) 决定（默认 train_pretrain → pre_train）。
#
# 可选: TRAIN_TARGET, TRAIN_TARGET_DEFAULT, TRAIN_SCRIPT
#
# 执行示例（建议在 `/minilm` 下执行）：
# ./scripts/run_pretrain_1gpu.sh
#   # 如需指定配置文件：
# ./scripts/run_pretrain_1gpu.sh \
#     --train_args_file config/pretrain/train_args.json \
#     --data_config_file config/pretrain/data_config.json
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec uv run -- python -m "${TRAIN_MODULE}" "$@"
