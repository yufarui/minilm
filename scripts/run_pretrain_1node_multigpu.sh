#!/usr/bin/env bash
# 单机多卡 torchrun。train_type 由入口 Python 模块决定，不在 shell 传参。
#
# 环境变量: NPROC_PER_NODE（默认 2）; 可选 TRAIN_TARGET, TRAIN_SCRIPT
#
# 执行示例（建议在 `/minilm` 下执行）：
#   NPROC_PER_NODE=8 SWANLAB_PROJECT=xxx SWANLAB_MODE=xxx SWANLAB_API_KEY=xxx ./scripts/run_pretrain_1node_multigpu.sh
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
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
source "${ROOT}/scripts/include/swanlab_env.sh"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m "${TRAIN_MODULE}" "$@"
