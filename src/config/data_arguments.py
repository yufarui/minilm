from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainDataArguments:
    """训练阶段共享数据参数（训练/验证路径、分词器、通用诊断）。"""

    model_config_file: Optional[str] = field(
        default="config/config.json",
        metadata={
            "help": "模型配置 JSON"
        },
    )
    train_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "训练集路径（JSON/JSONL，供 datasets load_dataset data_files）"
        },
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "验证或测试集路径；为空且 TrainingArguments.do_eval 时需自行关闭评估或补全路径"
        },
    )
    eval_domains_json: Optional[str] = field(
        default=None,
        metadata={
            "help": "分领域验证：指向「领域清单」JSON 文件路径（不是把所有领域混在一个数据文件里）。"
            "该 JSON 顶层为对象：键=领域名，值=该领域单独的 JSONL 数据路径。"
            "可与 eval_data_path 同时使用，此时主验证集会登记为键 eval。"
            "Trainer 将分别报告 eval_<领域名>_loss。示例见 config/pretrain/eval_domains.example.json。"
        },
    )
    tokenizer_name_or_path: str = field(
        default="tokenizer/minilm",
        metadata={"help": "Hugging Face 分词器名或本地目录"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "SFT 等场景的截断长度；预训练若未单独设置 pack_bin_size 则 packing 包长回退为此值"
        },
    )
    pack_bin_size: int | None = field(
        default=8192,
        metadata={
            "help": "打包时每个 bin 最大 token 数；SFT 入口通常使用 max_seq_length。"
        },
    )
    diag_every_n_steps: int = field(
        default=0,
        metadata={
            "help": "TrainingDiagnosticsCallback：每隔多少 global_step 在验证集上算 next-token top-1 与熵；0 关闭"
        },
    )
    diag_num_eval_batches: int = field(
        default=1,
        metadata={"help": "诊断回调在验证集上最多前向的 micro-batch 数"},
    )
    diag_gen_every_n_steps: int = field(
        default=0,
        metadata={
            "help": "TrainingDiagnosticsCallback：每隔多少步用固定前缀做 greedy/sample 生成并打日志；0 关闭"
        },
    )
    diag_gen_max_new_tokens: int = field(
        default=64,
        metadata={"help": "诊断生成最大新 token 数"},
    )
    diag_gen_prompts_json: Optional[str] = field(
        default=None,
        metadata={
            "help": "诊断生成前缀 JSON；支持字符串数组，或 SFT 对话对象数组（conversations/tools）"
        },
    )
    diag_gen_do_sample: bool = field(
        default=False,
        metadata={"help": "诊断生成是否采样（否则 greedy）"},
    )
    diag_gen_temperature: float = field(
        default=0.8,
        metadata={"help": "diag_gen_do_sample=True 时的 temperature"},
    )


@dataclass
class PretrainDataArguments(TrainDataArguments):
    """预训练数据参数。"""

    pack_bin_size: int | None = field(
        default=8192,
        metadata={
            "help": "预训练 packing：每个 bin（拼接后单条）最大 token 数；None 时使用 max_seq_length；"
            "与 pack_bin_schedule 同时存在时，schedule 未覆盖的尾段或回退仍用此值"
        },
    )
    pack_bin_schedule: Optional[list] = field(
        default=None,
        metadata={
            "help": "预训练按「打包后样本下标」分阶段提升窗口：JSON 数组。"
            "每项为对象：前若干项含 until_index（该阶段结束的累计样本数，不含）与 pack_bin_size；"
            "最后一项仅需 pack_bin_size，表示直到语料耗尽。例："
            '[{"until_index":50000,"pack_bin_size":2048},{"until_index":200000,"pack_bin_size":4096},{"pack_bin_size":8192}]'
        },
    )
    pack_sort_order: str = field(
        default="file_order",
        metadata={
            "help": "已弃用：预训练 ``PreTrainDataset`` 固定按 JSONL 行顺序拼接 Token 流，"
            "不再排序；保留该字段仅为兼容旧配置，值会被忽略。"
        },
    )
    swanlab_diag_log_attn_every: Optional[int] = field(
        default=None,
        metadata={
            "help": "MiniLMSwanlabDiagCallback：每隔多少 global_step 做一次注意力探针（额外前向）；None 表示关闭"
        },
    )
    swanlab_diag_log_hidden_every: Optional[int] = field(
        default=None,
        metadata={
            "help": "MiniLMSwanlabDiagCallback：每隔多少步记录各层 hidden 统计；None 表示关闭"
        },
    )


@dataclass
class SftDataArguments(TrainDataArguments):
    """SFT 数据参数。"""

    pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "SFT：预训练完成后的 checkpoint 目录（含 config.json 与权重）；"
            "为空时仅从 model_config_file 初始化结构（随机权重，一般仅作调试）"
        },
    )
    diag_sft_tool_json_every_n_steps: int = field(
        default=0,
        metadata={
            "help": "SFT：每隔 N 步用前缀生成文本，从输出中抽取 JSON 子串并统计可解析比例；0 关闭"
        },
    )
    diag_sft_tool_json_max_new_tokens: int = field(
        default=128,
        metadata={"help": "SFT 工具 JSON 生成探针的最大新 token 数"},
    )
    diag_sft_tool_json_prompts_json: Optional[str] = field(
        default=None,
        metadata={"help": "SFT 工具 JSON 探针前缀 JSON；支持字符串数组，或 SFT 对话对象数组（conversations/tools）"},
    )


@dataclass
class DpoDataArguments(TrainDataArguments):
    """DPO 数据参数。"""

    pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "DPO：策略模型 checkpoint 目录（含 config.json 与权重）"
        },
    )
    ref_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "DPO：参考模型 checkpoint；为空时由 DPOTrainer 按默认逻辑处理参考分支"
        },
    )
    dpo_beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta 超参数"},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "DPOTrainer 最大 prompt 长度"},
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={"help": "DPOTrainer 截断方式：keep_start 或 keep_end"},
    )
