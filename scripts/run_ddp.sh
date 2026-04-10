#!/usr/bin/env bash
# 多卡 DDP（torchrun，单机多卡）。请在项目根目录执行。
# 多机训练请勿使用 --standalone，请按 PyTorch 文档配置 torchrun 的 node 与 rendezvous。
#
# 环境变量:
#   STAGE              同 run_single_gpu.sh，默认 pretrain
#   NPROC_PER_NODE     本机进程数（通常等于 GPU 数），默认 1
#   MASTER_ADDR        默认 127.0.0.1
#   MASTER_PORT        默认 29500
#
# 示例:
#   cd /path/to/minilm
#   STAGE=pretrain NPROC_PER_NODE=2 bash scripts/run_ddp.sh
#   STAGE=sft NPROC_PER_NODE=4 MASTER_PORT=29501 bash scripts/run_ddp.sh --max_steps 1000
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_train_resolve.sh
source "${SCRIPT_DIR}/_train_resolve.sh"

cd "${ROOT}"
STAGE="${STAGE:-pretrain}"
MOD="$(resolve_stage_to_module "${STAGE}")" || exit 1

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m "${MOD}" \
  "$@"
