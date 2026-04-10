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
| `tokenizer/minilm/` | 默认本地 tokenizer 目录 |
| `tests/` | 单元测试与数据集构造测试 |

## 典型工作流

1. **准备数据**：按语雀文档与 `data/sft/README.md` 等说明准备 JSONL（大文件建议勿提交仓库，见 `.gitignore`）。
2. **（可选）训练或更新 tokenizer**：`python -m src.tokenizer.train_tokenizer`（参数见脚本内说明）。
3. **预处理**：编写任务 YAML，运行：
   ```bash
   python -m src.preprocess.run_preprocess --config config/preprocess/pipeline.job.yaml
   python -m src.preprocess.run_preprocess --config config/preprocess/sft.pipeline.job.yaml
   ```
4. **预训练**：
   ```bash
   python -m src.trainer.train_pretrain
   ```
   训练参数与数据路径由 `TrainConfig` 从 `config/pretrain/train_args.json`、`config/pretrain/data_config.json` 等加载（可通过命令行覆盖，与 `TrainConfig` 实现一致）。
5. **SFT**：
   ```bash
   python -m src.trainer.train_full_sft
   ```
   配置位于 `config/sft/`；`pretrained_model_path` 需指向预训练导出目录。
6. **DPO**：
   ```bash
   python -m src.trainer.train_dpo
   ```
   配置位于 `config/dpo/`；需已具备策略模型 checkpoint。

具体超参、数据字段与排错建议以 **[语雀文档](https://www.yuque.com/jeffery-zmlmw/sv9x6g/xgqr9u1g7n5ses62#yIFpX)** 为准。

## 测试

```bash
# 建议显式带上 PYTHONPATH（部分环境依赖）
set PYTHONPATH=.
pytest tests/preprocess tests/dataset_test
```

预处理与数据集测试会写入 `tests/tmp/`（已在 `.gitignore` 中忽略）。

## 许可证

若仓库根目录未包含 `LICENSE` 文件，上传 GitHub 前请自行补充许可证声明。
