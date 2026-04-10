#!/usr/bin/env bash
# 从预训练 / SFT JSONL 汇总 tokenizer 训练用纯文本（调用 collect_tokenizer_corpus）。
# 请在项目根目录执行，或任意目录下传入脚本绝对路径。
#
# 无参数时：须至少通过环境变量或下方示例自行传入数据源（模块要求 --pretrain-jsonl 与/或 --sft-jsonl）。
# 所有参数透传给 Python 模块。
#
# 示例:
#   cd /path/to/minilm
#   bash scripts/collect_tokenizer_corpus.sh \
#     --pretrain-jsonl data/pretrain/pretrain_train.jsonl \
#     --tokenizer-path tokenizer/minilm \
#     --output-path tokenizer/minilm/train_tokenizer.txt \
#     --max-pretrain-rows 800000
#
#   bash scripts/collect_tokenizer_corpus.sh \
#     --pretrain-jsonl data/pretrain/pretrain_train.jsonl \
#     --sft-jsonl data/sft/train.jsonl \
#     --tokenizer-path tokenizer/minilm \
#     --output-path tokenizer/minilm/train_tokenizer.txt \
#     --max-pretrain-rows 500000 --max-sft-rows 200000 --seed 42
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

exec python -m src.tokenizer.collect_tokenizer_corpus "$@"
