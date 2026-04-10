# 数据准备说明（Pretrain / SFT / DPO）

本目录用于放训练数据与切分结果。建议仅提交小样例，正式大语料放对象存储或本地数据盘。

## 目录建议

- `data/pretrain/`：预训练语料（输入与切分后的 train/val）。
- `data/sft/`：SFT 对话语料（输入与切分后的 train/val）。
- `data/dpo/`：DPO 偏好对语料（train/val）。

## Pretrain 数据（`data/pretrain/`）

### 输入格式

- JSONL，每行至少包含：
  - `text`：字符串，原始文本。

示例：

```json
{"text":"这是第一段预训练语料。"}
{"text":"This is another pretrain sample."}
```

### 处理流程

1. 原始 JSONL 准备好后，在 `config/preprocess/pipeline.job.yaml` 配置输入输出路径。
2. 执行 `bash scripts/preprocess_pretrain.sh` 进行清洗、去重与切分。
3. 训练时 `config/pretrain/data_config.json` 指向清洗后的 `train_data_path` / `eval_data_path`。

## SFT 数据（`data/sft/`）

### 输入格式

- JSONL，每行至少包含：
  - `conversations`：消息数组，元素含 `role` 与 `content`。

示例：

```json
{"conversations":[{"role":"user","content":"介绍一下杭州"},{"role":"assistant","content":"杭州是浙江省省会..."}]}
```

可选字段：

- `tools`（通常挂在首条 system 消息里，支持数组或 JSON 字符串）。
- `assistant.tool_calls`（支持数组或 JSON 字符串，预处理可尝试修复）。

### 处理流程

1. 在 `config/preprocess/sft.pipeline.job.yaml` 配置输入输出路径。
2. 执行 `bash scripts/preprocess_sft.sh`，会做结构检查、清洗、去重，并可切分 tool/multi-turn 验证集。
3. 训练时 `config/sft/data_config.json` 指向 `train_data_path` / `eval_data_path`。

## DPO 数据（`data/dpo/`）

支持两种格式，任选其一：

### 格式 A：对话偏好对（推荐）

- 每行必须有 `chosen` / `rejected`，两者都是消息列表（最后一条应为 assistant）。

示例：

```json
{
  "chosen":[{"role":"user","content":"写一句鼓励的话"},{"role":"assistant","content":"你已经很棒了，继续前进。"}],
  "rejected":[{"role":"user","content":"写一句鼓励的话"},{"role":"assistant","content":"我不知道。"}]
}
```

### 格式 B：扁平三元组

- 每行必须有 `prompt`、`chosen`、`rejected`，且均为字符串。

示例：

```json
{"prompt":"写一句鼓励的话","chosen":"你已经很棒了，继续前进。","rejected":"我不知道。"}
```

### 使用说明

- DPO 训练配置在 `config/dpo/data_config.json`。
- 若使用对话偏好对格式，训练时会通过 tokenizer/chat template 生成 prompt。

## 最小落地检查清单

- 路径存在：`train_data_path` / `eval_data_path` 均可访问。
- 字段正确：pretrain 有 `text`；sft 有 `conversations`；dpo 满足两种格式之一。
- 编码正确：统一 UTF-8，JSONL 一行一个 JSON 对象。
- 样本可解析：无空行、无截断 JSON、无超大脏字段。
