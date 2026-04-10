# shellcheck shell=bash
# 由 run_pretrain_*.sh 在设置 ROOT 并 cd 到项目根之后 source。
#
# train_type 由各 Python 入口内写死传入 TrainConfig.load_configs，不在此注入 CLI。
#
# 可选环境变量:
#   TRAIN_TARGET        入口短名，默认 train_pretrain → 模块 src.trainer.<TRAIN_TARGET>
#   TRAIN_TARGET_DEFAULT 由启动脚本设置时可作为 TRAIN_TARGET 的默认值
#   TRAIN_SCRIPT        相对项目根的 .py；设置则覆盖默认 src/trainer/<TRAIN_TARGET>.py
#
# 导出:
#   TRAIN_MODULE=src.trainer.<TRAIN_TARGET>
#   TRAIN_SCRIPT_REL    供 deepspeed 执行的相对路径 .py
#
: "${ROOT:?ROOT must be set before sourcing train_stage.inc.sh}"

TRAIN_TARGET="${TRAIN_TARGET:-${TRAIN_TARGET_DEFAULT:-train_pretrain}}"
TRAIN_MODULE="src.trainer.${TRAIN_TARGET}"
if [[ -n "${TRAIN_SCRIPT:-}" ]]; then
  TRAIN_SCRIPT_REL="${TRAIN_SCRIPT}"
else
  TRAIN_SCRIPT_REL="src/trainer/${TRAIN_TARGET}.py"
fi
