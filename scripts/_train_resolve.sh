#!/usr/bin/env bash
# shellcheck shell=bash
# 由 run_single_gpu.sh / run_ddp.sh / run_deepspeed.sh source，勿单独执行。

resolve_stage_to_module() {
  case "${1:-}" in
    pretrain | pre_train)
      printf '%s' "src.trainer.train_pretrain"
      ;;
    sft)
      printf '%s' "src.trainer.train_full_sft"
      ;;
    dpo)
      printf '%s' "src.trainer.train_dpo"
      ;;
    *)
      echo "未知 STAGE=${1:-}，可选: pretrain | sft | dpo" >&2
      return 1
      ;;
  esac
}
