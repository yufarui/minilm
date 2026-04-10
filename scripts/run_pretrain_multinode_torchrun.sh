#!/usr/bin/env bash
# 多机多卡 torchrun。train_type 由入口 Python 模块决定。
#
# 必填: MASTER_ADDR, NNODES, NODE_RANK, NPROC_PER_NODE
# 可选: MASTER_PORT（默认 29500）, TRAIN_TARGET, TRAIN_SCRIPT
#
# 执行示例（建议在 `/minilm` 下执行；各节点 NODE_RANK 需不同）：
#   MASTER_ADDR=10.0.0.1 NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
#     SWANLAB_PROJECT=xxx SWANLAB_MODE=xxx SWANLAB_API_KEY=xxx \
#     ./scripts/run_pretrain_multinode_torchrun.sh
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
: "${MASTER_ADDR:?请设置 MASTER_ADDR}"
: "${NNODES:?请设置 NNODES}"
: "${NODE_RANK:?请设置 NODE_RANK}"
: "${NPROC_PER_NODE:?请设置 NPROC_PER_NODE}"
MASTER_PORT="${MASTER_PORT:-29500}"
source "${ROOT}/scripts/include/swanlab_env.sh"
exec torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m "${TRAIN_MODULE}" \
  "$@"
