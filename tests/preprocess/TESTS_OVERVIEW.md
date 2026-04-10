# Preprocess 测试说明

本文档记录 `tests/preprocess` 下测试覆盖范围、伪造数据策略，以及当前已知环境问题。

## 测试文件与覆盖功能

### `test_preprocess_pipelines.py`

- **覆盖对象**
  - `src.preprocess.strategies.pipeline.PreprocessPipeline`（预训练路径）
  - `src.preprocess.strategies.sft_pipeline.SftPreprocessPipeline`（SFT 路径）
- **测试点**
  - 预训练路径：
    - `basic clean` 过滤（长度不足样本剔除）
    - `exact dedup` 去重行为
    - 关闭 `near_dedup` 时的阶段统计行为
    - 输出条数与去重后唯一性校验
  - SFT 路径：
    - 空对话过滤
    - 角色链校验（`strict_role_order`）
    - 拒答过滤（`filter_refuse_replies`）
    - 文本长度过滤
    - marker 归一化计数（`normalize_markers`）
    - `tool_calls` 修复计数（`repair_tool_calls`）
    - `exact dedup` 去重与最终输出数量

### `test_text_quality.py`

- **覆盖对象**
  - `src.preprocess.text_quality.normalize`
  - `src.preprocess.text_quality.length`
  - `src.preprocess.text_quality.symbol_ratio`
  - `src.preprocess.text_quality.language`
  - `src.preprocess.text_quality.tokens`
  - `src.preprocess.text_quality.pipeline.apply_text_quality`
- **测试点**
  - `normalize_text`：NFKC、换行统一、行尾空白处理
  - `apply_char_length_bounds`：短文本拒绝、超长截断
  - `passes_symbol_ratio_checks`：不可打印字符比例、标点比例阈值
  - `lang_matches / resolve_allowed_langs / is_language_allowed`：语言匹配逻辑与后端参数校验
  - `needs_tokenizer / passes_token_bounds / top_token_entries`：token 上下界与统计输出
  - `apply_text_quality`：完整拒绝原因分支
    - `empty`
    - `length`
    - `non_printable`
    - `punctuation`
    - `language`
    - `tokens`

## 伪造数据策略（不使用 `data` 目录）

- **预训练场景**：构造 100 条 JSONL
  - 20 条长度不足
  - 20 条可通过清洗但存在重复（10 组重复）
  - 60 条有效唯一文本
- **SFT 场景**：构造 100 条 JSONL
  - 10 条空对话
  - 10 条角色顺序错误
  - 10 条拒答样本
  - 10 条长度不足样本
  - 20 条重复对话（10 组）
  - 40 条有效唯一对话（含 1 条可修复 `tool_calls` 字符串）

## 测试执行结果

- 命令：
  - `PYTHONPATH='.' pytest tests/preprocess/test_preprocess_pipelines.py tests/preprocess/test_text_quality.py`
- 结果：
  - `8 passed`

## 当前环境问题说明

### 1) `src/ref_model` 相关模型加载受 `torchvision` 兼容影响

- 现象：
  - 调用 `transformers` 加载 `AutoModelForCausalLM` / `AutoModelForMaskedLM` 时，触发 `torchvision` 导入链报错：
    - `RuntimeError: operator torchvision::nms does not exist`
  - 进一步导致 `ModuleNotFoundError`（如 `GPT2LMHeadModel` / `PreTrainedModel`）的级联异常。
- 影响：
  - `src/ref_model.causal_lm`、`src/ref_model.masked_lm`、`src/ref_model.sentence_transformer` 在当前环境无法稳定完成加载验证。

### 2) 网络超时导致 HuggingFace 模型下载不稳定

- 现象：
  - 从 HF Hub 拉取模型时出现 `ReadTimeoutError`、重试告警。
- 影响：
  - 即使代码路径正确，参考模型初始化也可能因网络问题失败。

### 3) `fasttext` 安装说明

- 已安装 `fasttext-wheel==0.9.2`（Windows wheel 方案）
- 安装位置：
  - `D:\env\python311\Lib\site-packages`
- 说明：
  - 该安装仅解决 fasttext 依赖，不会解决 `torchvision` 与 `transformers` 的兼容问题。

## 建议后续动作

- 优先对齐 `torch` 与 `torchvision` 版本，确保 `torchvision::nms` 注册正常。
- 网络环境允许时，预下载并缓存 `src/ref_model` 所需 HF 模型，再执行 CPU smoke test。
