# MiniLM

小型因果语言模型训练与数据流水线：预训练（CPT）、全量 SFT、DPO 偏好对齐，配套自定义 tokenizer、JSONL 预处理与数据集打包。

**更完整的「从零开始的训练」说明、背景与流程拆解，见作者语雀文档：**

[从零开始的训练介绍 · 语雀](https://www.yuque.com/jeffery-zmlmw/sv9x6g/xgqr9u1g7n5ses62#yIFpX)

本 README 侧重仓库内**目录结构、安装与命令入口**，与上述文档互为补充。

## 功能概览

- **模型**：`MiniLmForCausalLM` / `MiniLMModel`（MoE 等选项见 `config/config.json`）
- **Tokenizer**：项目内 `tokenizer/minilm`（可由 `src/tokenizer/train_tokenizer.py` 训练或对齐）
- **数据预处理**：预训练 / SFT 共用 CLI，任务由 YAML 描述（`kind: pretrain | sft`）
- **数据集**：`PreTrainDataset`（流式 packing）、`SFTDataset`（对话 + packing）、`DPODataset`（偏好对，兼容对话 / 扁平格式）
- **训练脚本**：`train_pretrain`、`train_full_sft`、`train_dpo`（基于 Hugging Face `Trainer` / TRL `DPOTrainer`）

## 环境要求

- Python **≥ 3.11**
- PyTorch、Transformers、TRL 等见 [`pyproject.toml`](pyproject.toml)

## 安装

```bash
# 可编辑安装（开发）
pip install -e .

# 可选：DeepSpeed
pip install -e ".[deepspeed]"

# 可选：完整预处理依赖（近似去重、主题审计、fasttext 等）
pip install -e ".[preprocess-full]"
```

使用 [uv](https://github.com/astral-sh/uv) 时示例：

```bash
uv pip install -e ".[deepspeed,preprocess-full]"
```

> **说明**：`fasttext` 的语言模型权重（如 `lid.176.bin`）不会随 pip 自动下载，需在配置中设置 `fasttext_model_path`（参见 `config/preprocess/sft.pipeline.job.yaml` 注释）。

## 仓库结构（简要）

| 路径 | 说明 |
|------|------|
| `src/model/` | 模型定义 |
| `src/dataset/` | 预训练 / SFT / DPO 数据集 |
| `src/trainer/` | 训练入口 |
| `src/preprocess/` | 预处理流水线与 CLI |
| `src/tokenizer/` | 分词器训练与语料收集 |
| `config/` | 模型、预训练 / SFT / DPO 的 `train_args`、`data_config` 等 |
| `scripts/` | Bash 包装：训练、DeepSpeed/DDP、tokenizer 语料与训练、预处理等（见下节） |
| `tokenizer/minilm/` | 默认本地 tokenizer 目录 |
| `tests/` | 单元测试与数据集构造测试 |

## `scripts/` 入口脚本

| 脚本 | 作用 |
|------|------|
| `run_single_gpu.sh` | 单 GPU 训练；环境变量 `STAGE` 为 `pretrain`、`sft` 或 `dpo`，其余参数透传给 Trainer。 |
| `run_ddp.sh` | 单机多卡 DDP（`torchrun`）；`NPROC_PER_NODE` 等见脚本注释。 |
| `run_deepspeed.sh` | DeepSpeed 启动训练；`NUM_GPUS`、`DEEPSPEED_CONFIG` 等见脚本注释。 |
| `collect_tokenizer_corpus.sh` | 从预训练 / SFT JSONL 汇总 `train_tokenizer.txt`；参数透传至 `src.tokenizer.collect_tokenizer_corpus`。 |
| `train_tokenizer.sh` | 调用 `src.tokenizer.train_tokenizer` 训练 BBPE 并导出 HF tokenizer（当前 `main()` 内路径与超参见源码）。 |
| `preprocess_pretrain.sh` | 预训练预处理；无参时默认 `--config config/preprocess/pipeline.job.yaml`，有参则原样交给 `run_preprocess`。 |
| `preprocess_sft.sh` | SFT 预处理；无参时默认 `config/preprocess/sft.pipeline.job.yaml`。可用 `PREPROCESS_CONFIG` 覆盖默认 YAML。 |
| `render_minilm_arch.py` | 绘制 MiniLM 块结构示意图（matplotlib；用法见文件内说明）。 |
| `_train_resolve.sh` | 被 `run_*.sh` source，将 `STAGE` 解析为 Python 模块名；勿单独执行。 |

## 核心模块说明

### Tokenizer（`src/tokenizer/`）

- `collect_tokenizer_corpus.py`：从预训练 `text` 与 SFT `conversations` 汇总 tokenizer 训练语料，写出 `tokenizer/minilm/train_tokenizer.txt`。
- `train_tokenizer.py`：训练 BBPE 并导出 Hugging Face Fast tokenizer（包含 chat template 与特殊 token 对齐）。
- 实践上可先跑 `scripts/collect_tokenizer_corpus.sh`，再跑 `scripts/train_tokenizer.sh`。

### 模型结构（`src/model/`）

- 主干为 `MiniLMModel`：`Embedding -> N x DecoderLayer -> RMSNorm`。
- `MiniLmForCausalLM` 在主干后接 `lm_head` 输出 logits，并在训练期把 MoE `aux_loss` 加到总 loss。
- 注意力位置编码使用 RoPE（见 `rotary_embedding.py`/`rope_yarn.py`），生成时支持 KV cache。
- 具体结构参数（层数、hidden、heads、MoE）由 `config/config.json` 控制。

### 数据清洗（`src/preprocess/`）

- 统一入口：`run_preprocess.py`，通过任务配置里的 `kind: pretrain | sft` 选择清洗策略。
- pretrain 流水线（`strategies/pipeline.py`）：基础质量过滤 -> 精确去重 -> 近似去重 -> 可选 PPL 统计/过滤 -> 统计输出。
- sft 流水线（`strategies/sft_pipeline.py`）：对话结构检查、tool_calls 修复、拒答过滤、文本质量过滤、去重与诊断统计。
- 数据准备规范见 `data/README.md`（含 pretrain/sft/dpo 字段与格式要求）。

### 数据集加载（`src/dataset/`）

- `PreTrainDataset`：按文件顺序把文档拼成 token 流，文档间插 `<|endoftext|>`，再按 `pack_bin_size`/`pack_bin_schedule` 切块。
- `SFTDataset`：将 `conversations` 经 chat template 编码，`labels` 仅监督 assistant 段，并做 first-fit packing。
- `DPODataset`：支持 chat 偏好对与扁平三元组两种输入，统一输出 DPO 训练所需的 `prompt/chosen/rejected`。

### 掩码处理（`src/util/data_collator.py`）

- collator 统一做动态 padding，输出 `input_ids`、`labels`、`position_ids`、`attention_mask`。
- `position_ids` 在 packing 分隔符 `<|endoftext|>` 处重置，实现“段内从 0 重新计数”。
- attention mask 由 `labels` 切成 prefix/causal block：prefix 不计 loss 且可被后续监督段看到；causal 段为下三角自回归可见。

### 监控指标（`src/monitor/`）

- `monitor/common`：loss 归一化、clip 后 grad norm、top-1/熵诊断、固定 prompt 生成探针。
- `monitor/pretrain`：预训练专用回调（含 SwanLab MoE/隐藏层相关诊断）。
- `monitor/sft`：SFT 分项 loss 镜像与工具 JSON 生成可解析率探针。
- 监控开关与频率主要由各阶段 `data_config.json` 中 `diag_*` 字段控制。

## 最短实战路径附录

以下以单机为例（Bash）：

```bash
# 0) 安装
uv sync --all-extras

pip install -e ".[deepspeed,preprocess-full]"

# 1) 数据准备（先按 data/README.md 放好原始 JSONL）

# 2) tokenizer（可选：若要重训 tokenizer）
bash scripts/collect_tokenizer_corpus.sh --pretrain-jsonl data/pretrain/pretrain_t2t.jsonl --sft-jsonl data/sft/sft_t2t_mini.jsonl
bash scripts/train_tokenizer.sh

# 3) 清洗与切分
bash scripts/preprocess_pretrain.sh
bash scripts/preprocess_sft.sh

# 4) 预训练
STAGE=pretrain bash scripts/run_single_gpu.sh

# 5) SFT（依赖 pretrain checkpoint）
STAGE=sft bash scripts/run_single_gpu.sh

# 6) DPO（依赖 sft checkpoint）
STAGE=dpo bash scripts/run_single_gpu.sh
```

如需多卡，替换为 `scripts/run_ddp.sh` 或 `scripts/run_deepspeed.sh`。

## 测试

```bash
# 建议显式带上 PYTHONPATH（部分环境依赖）
set PYTHONPATH=.
pytest tests/preprocess tests/dataset_test
```

预处理与数据集测试会写入 `tests/tmp/`（已在 `.gitignore` 中忽略）。

## 许可证

本项目采用 **Apache-2.0** 许可证。完整条款见仓库根目录 `LICENSE` 文件。
