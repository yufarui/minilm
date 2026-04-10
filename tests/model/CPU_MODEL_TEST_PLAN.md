# MiniLM `src/model` 测试计划（CPU 主跑 + GPU 对照）

## 目标范围

本文档定义 `src/model` 下模块的功能测试方案。
主目标是在 CPU 上稳定验证张量形状、数据流、缓存行为和关键数值性质；同时保留一套 GPU 对照配置，确保后续可无缝扩展到 CUDA 环境。

## 测试环境与配置基线

- 配置基线：
  - 默认模型结构 **开启 MoE**，并与 `config/config.json` 对齐：
    - `moe_enable=true`
    - `n_routed_experts=4`
    - `n_shared_experts=1`
    - `num_experts_per_tok=1`
    - `scoring_func="sigmoid"`
    - `aux_loss_alpha=0.01`
    - `norm_topk_prob=true`
    - `seq_aux=true`
- 词表规模：
  - `vocab_size` 不在测试中硬编码小值，默认与 `tokenizer/minilm/tokenizer_config.json` 对齐（当前为 `16000`），尽量不影响 tokenizer 相关配置。
- CPU 运行配置（主执行）：
  - 设备：CPU（`torch.device("cpu")`）
  - 精度：`float32`
  - 可复现性：
    - `torch.manual_seed(42)`
    - 大部分测试关闭 dropout（`config.attention_dropout = 0.0`，`model.eval()`）
- GPU 对照配置（保留，不作为本阶段强制门禁）：
  - 设备：CUDA（可用时）
  - 精度：`float16`/`bfloat16`（按硬件能力）
  - 与 CPU 使用同一组测试逻辑，允许放宽数值容差
- 可复现性：
  - `torch.manual_seed(42)`
  - 大部分测试关闭 dropout（`config.attention_dropout = 0.0`，`model.eval()`）
- 为了 CPU 速度可缩小结构参数（如层数、hidden size），但 **不改变 MoE 默认开启和 tokenizer 对齐的 vocab 配置约束**。

## 优先级 1：核心基础模块

### 1) `RMSNorm`

- forward 后输入输出形状一致
- 输出 dtype 与输入 dtype 一致
- 输出不包含 `nan`/`inf`
- backward 可正常执行（输入和权重都有非空梯度）

### 2) `MLP`

- forward 形状：`[B, S, H] -> [B, S, H]`
- 能在 CPU 上处理小随机输入
- backward 能覆盖所有投影层权重

### 3) `RotaryEmbedding`

- `forward(x, position_ids)` 返回 `(cos, sin)`，且形状和 dtype 符合预期
- 支持 `batch > 1` 和可变序列长度
- 固定随机种子和输入时输出稳定、无非有限值
- 覆盖默认 rope 路径与 yarn rope 路径（`config.rope_scaling["type"] == "yarn"`）

### 4) `Attention.repeat_kv` 与 RoPE 辅助函数

- `repeat_kv` 头数扩展正确
- `rotate_half`、`apply_rotary_pos_emb` 形状保持正确
- CPU 上无 dtype/device 不匹配问题

## 优先级 2：注意力与解码层逻辑

### 5) `Attention.forward`（CPU 路径）

- 输出形状为 `[B, S, H]`
- eager attention 路径在 CPU + mask 场景可运行
- mask 行为基本正确：
  - 被屏蔽位置不会在容差外影响未屏蔽 token 输出
- 输出有限；eager 路径下 attention weights 也应有限

### 6) `DecoderLayer.forward`

- 残差连接后形状正确且输出有限
- 支持有/无 `attention_mask` 两种输入
- backward 能穿过 attention 与 mlp/moe 分支

## 优先级 3：MoE 逻辑

### 7) `MoeGate.forward`

- 返回 `topk_idx`、`topk_weight`、`aux_loss`，形状符合预期
- `topk_idx` 取值范围在 `[0, n_routed_experts)` 内
- 当 `norm_topk_prob=True` 且 `top_k>1` 时，每个 token 的 top-k 权重和约等于 1
- 覆盖 `scoring_func`：
  - `sigmoid` 路径
  - `softmax` 路径
- aux loss 行为：
  - 训练模式 + `alpha>0` 时为非负标量
  - eval 模式返回零值或近零值

### 8) `Moe.forward`

- 输出形状为 `[B, S, H]`
- routed experts 路径可处理稀疏 token-expert 分配
- 当 `n_shared_experts > 0` 时 shared experts 路径可运行
- `self.aux_loss` 被正确写入且为有限值

## 优先级 4：端到端模型接口

### 9) `MiniLMModel.forward`

- 输入 `input_ids` 时，输出 `last_hidden_state` 形状为 `[B, S, H]`
- 支持 `inputs_embeds` 路径
- `input_ids` 与 `inputs_embeds` 互斥校验生效（非法输入抛出预期错误）
- `use_cache=True` 时返回非空 cache 对象
- 提供 `past_key_values` 时，增量一步前向可正常执行

### 10) `MiniLmForCausalLM.forward`

- logits 形状为 `[B, S, vocab_size]`（或 `logits_to_keep` 切片后的形状）
- 训练模式 + labels 时返回有限 loss
- MoE 启用时，loss 包含 aux 分量
- eval 模式无需 labels，也可正常返回 logits
- 支持 `logits_to_keep` 为 int 以及 tensor/slice 类索引

## 优先级 4.5：推理层（Inference）专项

### 11) 增量解码与缓存一致性

- 逐 token 推理（带 `past_key_values`）与整段一次性前向在同一位置 logits 近似一致（关闭 dropout）
- `use_cache=True` 时，cache 长度随步数增长符合预期
- `position_ids` 在增量推理场景下单调递增，且与 cache 长度匹配

### 12) 推理接口稳定性

- `MiniLmForCausalLM` 在 `eval()` + `torch.no_grad()` 下多步调用不报错
- `logits_to_keep` 在推理路径下可正确裁剪最后若干步 logits
- attention mask 在推理场景（含 padding）下行为符合预期

## 优先级 4.8：`TrainDataCollator` 产物一致性（重点）

> 重点验证 `src/util/data_collator.py` 输出的 `position_ids` 与 `attention_mask`，因为这两者直接影响训练时 RoPE 与可见性边界。

### 13) `position_ids` 正确性

- **pack 分段重置**：当 `input_ids` 中出现 `pad_token_id`（作为 pack 分隔符）时，下一 token 的位置从 0 重新开始
- **分隔符自身计数**：分隔符位置保留当前段末尾计数（与实现注释一致），仅分隔符后重置
- **右侧 batch padding**：动态补齐产生的右侧 padding 位置，其 `position_ids` 必须为 0
- **空/短序列边界**：长度 0、长度 1、全分隔符序列等边界输入行为稳定

### 14) `attention_mask` 正确性

- **输出形状**：`attention_mask` 为 `[B, 1, S, S]`
- **padding 不可见**：被右侧补齐的 padding 位（`input_ids == pad_token_id`）在 key 维必须被完全屏蔽
- **预训练路径（full_lm）**：
  - `labels[non_pad] == input_ids[non_pad]` 时走因果掩码
  - 同一 pack 段内保持下三角可见
  - 不同 pack 段之间严格隔离（后段不可见前段）
- **SFT 前缀路径（prefix）**：
  - `labels == ignore_index` 的前缀块内部全互见
  - 监督块内部为因果可见
  - pack 边界同样必须阻断跨段可见性

### 15) 与模型输入契约对齐

- `collator` 输出的 `position_ids` 与 `attention_mask` 可直接喂给 `MiniLMModel/MiniLmForCausalLM` 完成一次前向
- 在 CPU 上前向不报 shape/device 错误
- 关键断言：
  - `position_ids.shape == input_ids.shape`
  - `attention_mask.shape[-2:] == (S, S)`
  - mask 值域仅包含 0/1

## 负向与异常处理测试

- `MoeGate` 中非法 `scoring_func` 抛出 `NotImplementedError`
- 非法的 `input_ids`/`inputs_embeds` 使用抛出 `ValueError`
- 错误的 mask 形状触发运行时异常（断言异常类型/信息可适度宽松）

## 建议测试文件结构

- `tests/model/test_rms_norm.py`
- `tests/model/test_mlp.py`
- `tests/model/test_rotary_embedding.py`
- `tests/model/test_attention.py`
- `tests/model/test_decode_layer.py`
- `tests/model/test_moe_gate.py`
- `tests/model/test_moe.py`
- `tests/model/test_model_forward.py`
- `tests/model/test_causal_lm_forward.py`
- `tests/model/test_inference_cache.py`
- `tests/model/test_inference_decode_step.py`
- `tests/model/test_data_collator_position_ids.py`
- `tests/model/test_data_collator_attention_mask.py`
- `tests/model/test_data_collator_model_integration.py`

## 本阶段暂不覆盖

- 吞吐/时延基准测试
- 超长序列（大上下文）压力测试
- GPU 专属 kernel 行为（FlashAttention/CUDA 正确性与性能）
- 分布式训练、混合精度、多机多卡行为
- 将 GPU 结果作为硬门禁的 CI 规则（本阶段先保留 GPU 配置与本地可选执行）

## 验收标准

- 所有 CPU 模型测试可在本地通过 `pytest`
- 不依赖 GPU 可用性
- GPU 环境可用时，可用同一测试集完成对照运行（非阻塞）
- 总体耗时保持开发友好（目标：几分钟量级）
- 失败信息具备可定位性（形状、dtype、有限值检查、cache 协议）

