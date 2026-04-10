"""预处理流水线可观测统计：固定字段，便于监控与落盘 JSON。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TopicAuditStats:
    """BERTopic 审计结果；未启用或跳过时的语义见 ``ran`` / ``skipped``。"""

    ran: bool = False
    """本次是否执行了主题建模（启用且有文本）。"""
    skipped: bool = False
    """流水线配置为关闭主题审计时为 True。"""
    n_docs: int | None = None
    """参与建模的文档数（截断至 max_docs 后）。"""
    n_topics: int | None = None
    """检测到的主题数（不含 outlier 时与库定义一致）。"""
    topic_info_rows: int | None = None
    """topic_info 表行数。"""
    output_dir: str | None = None
    """报表写出目录（绝对路径字符串）；未写出时为 None。"""
    visualize_error: str | None = None
    """可视化失败时的错误信息。"""
    error: str | None = None
    """无法建模时的简短原因，如 empty。"""


@dataclass
class PreprocessPipelineStats:
    """各阶段保留文档数及可选质量指标；与流水线阶段顺序一致。"""

    input_lines: int = 0
    """读入的非空 JSONL 行数。"""
    after_basic_clean: int = 0
    """通过基础清洗（含可选段内去重）后的条数。"""
    after_exact_dedup: int = 0
    """精确去重后条数；未启用精确去重时与上一阶段相同。"""
    after_near_dedup: int = 0
    """近似去重后条数；未启用时与上一阶段相同。"""
    after_ppl_filter: int = 0
    """PPL 相关步骤之后条数：仅观测 PPL 时与近似去重后相同；启用分位删除后与过滤后一致。"""
    output_lines: int = 0
    """写出到输出 JSONL 的最终行数（应与 after_ppl_filter 一致）。"""
    ppl_threshold_low: float | None = None
    """启用 PPL 过滤时，低分位对应的 PPL 阈值（低于且非结构化可丢）。"""
    ppl_threshold_high: float | None = None
    """启用 PPL 过滤时，高分位对应的 PPL 阈值（高于则丢）。"""
    ppl_would_remove_count: int | None = None
    """仅计算 PPL、未应用分位删除时，按当前分位规则**本会**剔除的条数；已应用删除或未算 PPL 时为 None。"""
    bert_mlm_mean_nll: float | None = None
    """启用 BERT MLM 子集打分时，子集平均 token NLL；未启用为 None。"""
    doc_length_hist_bins: list[float] = field(default_factory=list)
    """写出语料的文档长度分布直方图边界（字符数，len(text)）。"""
    doc_length_hist_counts: list[int] = field(default_factory=list)
    """与 ``doc_length_hist_bins`` 对应的分桶计数。"""
    ppl_hist_bins: list[float] = field(default_factory=list)
    """写出语料的 PPL 分布直方图边界（仅在计算了 PPL 时存在）。"""
    ppl_hist_counts: list[int] = field(default_factory=list)
    """与 ``ppl_hist_bins`` 对应的分桶计数。"""
    topic_audit: TopicAuditStats = field(default_factory=TopicAuditStats)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """可选：如 ``text_quality_rejects``、token 直方图等。"""

    def to_json_dict(self) -> dict[str, Any]:
        """嵌套 dataclass 一并转为可 ``json.dump`` 的结构。"""
        return asdict(self)

    def stage_count_series(self) -> list[tuple[str, int]]:
        """各阶段保留文档数（用于漏斗图 / 折线）。"""
        return [
            ("读入 JSONL", self.input_lines),
            ("基础清洗后", self.after_basic_clean),
            ("精确去重后", self.after_exact_dedup),
            ("近似去重后", self.after_near_dedup),
            ("PPL 过滤后", self.after_ppl_filter),
            ("写出", self.output_lines),
        ]

    def drop_counts(self) -> list[tuple[str, int]]:
        """各阶段剔除条数（非负）；与 stage 顺序对应。"""
        s = self
        return [
            ("基础清洗剔除", max(0, s.input_lines - s.after_basic_clean)),
            ("精确重复剔除", max(0, s.after_basic_clean - s.after_exact_dedup)),
            ("近似重复剔除", max(0, s.after_exact_dedup - s.after_near_dedup)),
            ("PPL 过滤剔除", max(0, s.after_near_dedup - s.after_ppl_filter)),
        ]

    def retention_rate(self) -> float:
        if self.input_lines <= 0:
            return 0.0
        return self.output_lines / self.input_lines


@dataclass
class SftPreprocessStats:
    """SFT 对话 JSONL 预处理与训练前诊断摘要。"""

    input_lines: int = 0
    output_lines: int = 0
    skipped_empty_conversations: int = 0
    skipped_role_order: int = 0
    skipped_language: int = 0
    skipped_length: int = 0
    skipped_non_printable: int = 0
    skipped_punctuation: int = 0
    skipped_tokens: int = 0
    skipped_refuse_reply: int = 0
    skipped_think_samples: int = 0
    skipped_tool_json: int = 0
    tool_calls_repaired: int = 0
    markers_normalized_rows: int = 0
    after_exact_dedup: int = 0
    after_near_dedup: int = 0
    mean_turns: float | None = None
    role_violation_examples: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """含长度/token 分桶、token id 直方图顶项等，由流水线填充。"""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
