#!/usr/bin/env bash
# 训练 / 更新 BBPE 并写出 Hugging Face Fast tokenizer（调用 train_tokenizer）。
# 请在项目根目录执行，或任意目录下传入脚本绝对路径。
#
# 当前 Python 入口 ``src.tokenizer.train_tokenizer`` 的 ``main()`` 使用仓库内固定路径与超参
#（见该文件）；若需改语料路径、词表大小等，请直接编辑 ``src/tokenizer/train_tokenizer.py`` 中
# ``main()``，或通过本脚本所在环境与后续若增加的 CLI 对接。
#
# 示例:
#   cd /path/to/minilm
#   bash scripts/train_tokenizer.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

exec python -m src.tokenizer.train_tokenizer "$@"
