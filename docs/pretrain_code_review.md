# 预训练相关代码审阅（`src/`）

本文基于当前仓库实现，列出预训练链路中**值得优先关注**的问题与改进方向，便于后续迭代。

---

## 1. 训练入口与评估

### 1.1 未接入验证集

`train_pretrain.run_pretrain` 只构建 `train_dataset`，**未**根据 `TrainingArguments.do_eval` 与 `DataArguments.eval_data_path` 构造 `eval_dataset`，`Trainer` 始终在未传 `eval_dataset` 的情况下创建。

- 配置里若将 `do_eval` 设为 `true` 且提供 `eval_data_path`，当前代码仍**不会**做验证，行为与配置不一致。
- **建议**：在 `do_eval` 为真且路径存在时构建 `PreTrainDataset` 并传入 `Trainer(eval_dataset=...)`；否则在启动时显式校验并报错/降级。

### 1.2 评估指标模块空置

`src/metrics/eval_metrics.py` 当前几乎为空；若启用 `compute_metrics`，缺少与因果 LM（perplexity、token 准确率等）对齐的实现。

- **建议**：恢复或实现与 `Trainer` 兼容的 `compute_metrics`，并在开启 `do_eval` 时挂到 `Trainer`。

---

## 2. 数据管线

### 2.1 全量 `max_length` 定长 padding

`PreTrainDataset` 对每条样本使用 `padding="max_length"`，短文本会产生大量 padding token。

- 训练仍会对整段序列做前向（见下文的 `attention_mask`），**算力与显存效率偏低**。
- **建议**：可变长 + `DataCollator` 动态 padding（或 packing / 文档级切分），并与 `max_seq_length`、词表 BOS/EOS 策略统一设计。

### 2.2 缺少 `attention_mask`（定长 batch）

在 `PreTrainDataCollator` 的「等长直接 stack」分支中，batch **不**包含 `attention_mask`。`MiniLMModel` / `Attention` 在 `attention_mask is None` 时不会屏蔽 padding 位置，pad token 仍会参与注意力计算；`labels` 上虽用 `-100` 忽略 CE，但**表示学习与梯度仍受 pad 干扰**。

- **建议**：即使定长 batch，也生成 `attention_mask`（有效位置为 1，pad 为 0），与 `tokenizer.pad_token_id` 一致。

### 2.3 `PreTrainDataCollator._dynamic_pad` 实现错误

在 `src/util/data_collator.py` 的 `_dynamic_pad` 循环内：

- 使用 `labels = f["labels"]` **覆盖了**外层的列表累加变量 `labels`，随后对 `labels.append(...)` 会在第二轮及以后出错；动态 padding 路径在长度不一致时**不可靠**。
- 同时 `pad_token_id` 写死为 `0`，若与 `tokenizer.pad_token_id` 不一致会导致 mask/标签错误。

- **建议**：循环内改用不同局部变量名（如 `lab`），`pad_token_id` 默认取自 tokenizer 或 `DataArguments`。

### 2.4 数据字段与健壮性

- 假设每条样本必有 `sample["text"]`，无键名校验或异常提示。
- `load_dataset("json", data_files=...)` 对超大语料可能一次性占用较多内存；超大规模预训练通常需要流式/分片或 IterableDataset。

---

## 3. 模型与损失（MoE）

### 3.1 MoE `aux_loss` 未并入 `Trainer` 使用的 `loss`

`MiniLmForCausalLM.forward` 中 `loss` 仅来自 `loss_function`（交叉熵）；`outputs.aux_loss` 虽从 `MiniLMModel` 传出并放入 `MiniLMCausalLMOutputWithPast`，但**未与 `loss` 相加**。

- 在 `moe_enable=True` 时，门控负载均衡等辅助损失**不会参与反向**，与常见 MoE 训练做法不一致，易导致路由坍塌等问题。
- **建议**：在 `forward` 中显式 `loss = ce_loss + aux_loss`（注意标量/形状与混合精度）；或与 Transformers 对多损失组件的约定对齐。

---

## 4. 训练与配置

### 4.1 随机初始化起点

当前为 `MiniLmForCausalLM(model_config)` 直接训练，无 `from_pretrained` 权重。若目标是「小模型从零预训练」则合理；若期望热启动或继续训练，需在文档与脚本中区分流程。

### 4.2 `train_type` 仅代码内指定

`TrainConfig.load_configs(train_type, ...)` 由入口模块常量传入，**CLI 不可改阶段**。多阶段时依赖不同 Python 入口，需在文档中写清，避免与「命令行切阶段」的预期混淆。

### 4.3 SwanLab 诊断回调

`MiniLMSwanlabDiagCallback` 在开启 `log_attn_every` / `log_hidden_every` 时会**额外前向**，且依赖 `output_attentions` 等，对吞吐有影响；MoE 关闭时部分指标为空属正常，但回调仍会注册，可接受。

---

## 5. 小结（优先级建议）

| 优先级 | 主题 |
|--------|------|
| 高 | MoE 开启时 **将 `aux_loss` 并入总 loss** |
| 高 | **验证集 / `do_eval` 与 `Trainer` 行为一致** |
| 高 | **`PreTrainDataCollator._dynamic_pad` 变量遮蔽 bug** 与 `pad_token_id` 对齐 tokenizer |
| 中 | 定长 batch 仍传 **`attention_mask`**，避免 pad 参与注意力 |
| 中 | 补全 **`eval_metrics`** 与 `compute_metrics`（若做 eval） |
| 低 | 数据字段校验、大数据集加载策略、padding/序列策略优化 |

以上条目随业务目标（是否 MoE、是否必须 eval、数据规模）可调整优先级；审阅基准为当前 `src/trainer/train_pretrain.py`、`src/dataset/pre_train_dataset.py`、`src/util/data_collator.py`、`src/model/model.py` 及配置加载路径。
