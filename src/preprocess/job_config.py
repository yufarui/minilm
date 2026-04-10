"""从 YAML（或标准 JSON）任务文件加载预处理路径与流水线配置（替代仅靠 CLI --input/--output）。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from src.preprocess.basic_clean import BasicCleanConfig
from src.preprocess.deduplicate import ExactDedupConfig, NearDedupConfig
from src.preprocess.scoring import BertMlmNllConfig, Gpt2PplConfig
from src.preprocess.strategies.pipeline import PreprocessPipelineConfig
from src.preprocess.strategies.sft_pipeline import SftPipelineConfig
from src.preprocess.topic_audit import TopicAuditConfig
from src.util.path_util import resolve_under_project


@dataclass
class PretrainSplitConfig:
    enabled: bool = True
    val_size: int = 5000
    val_output_path: str | None = None
    seed: int = 42


@dataclass
class SftSplitConfig:
    enabled: bool = True
    tool_call_val_size: int = 1000
    multi_turn_val_size: int = 5000
    tool_call_val_output_path: str | None = None
    multi_turn_val_output_path: str | None = None
    seed: int = 42


@dataclass
class PreprocessJobFile:
    """
    任务 YAML 顶层字段：

    - ``kind``: ``pretrain``（默认）或 ``sft``，决定使用 ``pipeline`` 还是 ``sft`` 嵌套配置
    - ``input``: 输入 JSONL **文件路径**（相对项目根或绝对路径，见 ``resolve_under_project``）
    - ``output``: 输出 JSONL **文件路径**（同上）
    - ``stats_path``: 可选，统计结果 JSON **文件路径**
    - ``write_plots``: 是否生成统计图（默认 true；``kind: sft`` 时通常无预训练同款图表，可关）
    - ``plots_dir``: 图表输出目录；为空且 ``write_plots`` 时，默认为输出文件同目录下 ``<stem>_report/``
    - ``pipeline``: 预训练：嵌套字段覆盖 ``PreprocessPipelineConfig``
    - ``sft`` / ``sft_pipeline``: SFT：嵌套字段覆盖 ``SftPipelineConfig``
    - ``split``: 训练/验证拆分配置（默认开启）
      - ``split.pretrain.val_size`` 默认 5000
      - ``split.sft.tool_call_val_size`` 默认 1000
      - ``split.sft.multi_turn_val_size`` 默认 5000

    使用 ``yaml.safe_load`` 解析；无注释的标准 JSON 也可作为输入。
    """

    input_path: str
    output_path: str
    kind: str = "pretrain"
    stats_path: str | None = None
    write_plots: bool = True
    plots_dir: str | None = None
    pipeline: PreprocessPipelineConfig = field(default_factory=PreprocessPipelineConfig)
    sft: SftPipelineConfig = field(default_factory=SftPipelineConfig)
    pretrain_split: PretrainSplitConfig = field(default_factory=PretrainSplitConfig)
    sft_split: SftSplitConfig = field(default_factory=SftSplitConfig)

    @classmethod
    def load(cls, config_path: str | Path) -> PreprocessJobFile:
        p = resolve_under_project(Path(config_path))
        with p.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise TypeError("任务配置须为 YAML/JSON 映射（顶层对象）")
        raw: dict[str, Any] = loaded

        if "input" not in raw or "output" not in raw:
            raise ValueError("任务配置须包含顶层键 input 与 output")
        inp, out = raw["input"], raw["output"]
        if inp is None or out is None or not str(inp).strip() or not str(out).strip():
            raise ValueError("input 与 output 必须为非空字符串")

        kind = str(raw.get("kind", "pretrain")).strip().lower()
        if kind not in ("pretrain", "sft"):
            raise ValueError("kind 须为 pretrain 或 sft")

        pipe_dict = raw.get("pipeline")
        if pipe_dict is not None and not isinstance(pipe_dict, dict):
            raise TypeError("pipeline 必须为 YAML/JSON 映射")
        pipeline = _pipeline_config_from_dict(pipe_dict or {})

        sft_dict = raw.get("sft")
        if sft_dict is None:
            sft_dict = raw.get("sft_pipeline")
        if sft_dict is not None and not isinstance(sft_dict, dict):
            raise TypeError("sft / sft_pipeline 必须为 YAML/JSON 映射")
        sft_cfg = _sft_pipeline_config_from_dict(sft_dict or {})
        split_dict = raw.get("split")
        if split_dict is not None and not isinstance(split_dict, dict):
            raise TypeError("split 必须为 YAML/JSON 映射")
        pre_split, sft_split = _split_config_from_dict(split_dict or {})

        return cls(
            input_path=str(inp),
            output_path=str(out),
            kind=kind,
            stats_path=raw.get("stats_path"),
            write_plots=bool(raw.get("write_plots", True)),
            plots_dir=raw.get("plots_dir"),
            pipeline=pipeline,
            sft=sft_cfg,
            pretrain_split=pre_split,
            sft_split=sft_split,
        )

    def resolved_input(self) -> Path:
        return resolve_under_project(self.input_path)

    def resolved_output(self) -> Path:
        return resolve_under_project(self.output_path)

    def resolved_stats_path(self) -> Path | None:
        if not self.stats_path:
            return None
        return resolve_under_project(self.stats_path)

    def resolved_plots_dir(self) -> Path | None:
        if not self.write_plots:
            return None
        if self.plots_dir:
            return resolve_under_project(self.plots_dir)
        outp = self.resolved_output()
        return outp.parent / f"{outp.stem}_report"

    def pipeline_for_run(self) -> PreprocessPipelineConfig:
        """任务级 ``stats_path`` 覆盖 ``pipeline.stats_path``；路径按项目根解析。"""
        p = self.pipeline
        if self.stats_path:
            p = replace(p, stats_path=str(resolve_under_project(self.stats_path)))
        elif p.stats_path:
            p = replace(p, stats_path=str(resolve_under_project(p.stats_path)))
        return p

    def sft_pipeline_for_run(self) -> SftPipelineConfig:
        """任务级 ``stats_path`` 覆盖 ``sft.stats_path``。"""
        s = self.sft
        if self.stats_path:
            s = replace(s, stats_path=str(resolve_under_project(self.stats_path)))
        elif s.stats_path:
            s = replace(s, stats_path=str(resolve_under_project(s.stats_path)))
        return s

    def resolved_pretrain_val_output(self) -> Path:
        if self.pretrain_split.val_output_path:
            return resolve_under_project(self.pretrain_split.val_output_path)
        outp = self.resolved_output()
        return outp.parent / f"{outp.stem}_val{outp.suffix or '.jsonl'}"

    def resolved_sft_tool_val_output(self) -> Path:
        if self.sft_split.tool_call_val_output_path:
            return resolve_under_project(self.sft_split.tool_call_val_output_path)
        outp = self.resolved_output()
        return outp.parent / f"{outp.stem}_tool_val{outp.suffix or '.jsonl'}"

    def resolved_sft_multi_turn_val_output(self) -> Path:
        if self.sft_split.multi_turn_val_output_path:
            return resolve_under_project(self.sft_split.multi_turn_val_output_path)
        outp = self.resolved_output()
        return outp.parent / f"{outp.stem}_multi_turn_val{outp.suffix or '.jsonl'}"


def _split_config_from_dict(d: dict[str, Any]) -> tuple[PretrainSplitConfig, SftSplitConfig]:
    pre = PretrainSplitConfig()
    sft = SftSplitConfig()
    pd = d.get("pretrain")
    if isinstance(pd, dict):
        if "enabled" in pd:
            pre = replace(pre, enabled=bool(pd["enabled"]))
        if "val_size" in pd:
            pre = replace(pre, val_size=int(pd["val_size"]))
        if "val_output_path" in pd:
            v = pd["val_output_path"]
            pre = replace(pre, val_output_path=None if v is None else str(v))
        if "seed" in pd:
            pre = replace(pre, seed=int(pd["seed"]))
    sd = d.get("sft")
    if isinstance(sd, dict):
        if "enabled" in sd:
            sft = replace(sft, enabled=bool(sd["enabled"]))
        if "tool_call_val_size" in sd:
            sft = replace(sft, tool_call_val_size=int(sd["tool_call_val_size"]))
        if "multi_turn_val_size" in sd:
            sft = replace(sft, multi_turn_val_size=int(sd["multi_turn_val_size"]))
        if "tool_call_val_output_path" in sd:
            v = sd["tool_call_val_output_path"]
            sft = replace(sft, tool_call_val_output_path=None if v is None else str(v))
        if "multi_turn_val_output_path" in sd:
            v = sd["multi_turn_val_output_path"]
            sft = replace(sft, multi_turn_val_output_path=None if v is None else str(v))
        if "seed" in sd:
            sft = replace(sft, seed=int(sd["seed"]))
    return pre, sft


def _sft_pipeline_config_from_dict(d: dict[str, Any]) -> SftPipelineConfig:
    cfg = SftPipelineConfig()
    if "conversations_field" in d:
        cfg = replace(cfg, conversations_field=str(d["conversations_field"]))
    if "strict_role_order" in d:
        cfg = replace(cfg, strict_role_order=bool(d["strict_role_order"]))
    if "repair_tool_calls" in d:
        cfg = replace(cfg, repair_tool_calls=bool(d["repair_tool_calls"]))
    if isinstance(d.get("normalize_markers"), dict):
        cfg = replace(cfg, normalize_markers={str(k): str(v) for k, v in d["normalize_markers"].items()})
    if "filter_refuse_replies" in d:
        cfg = replace(cfg, filter_refuse_replies=bool(d["filter_refuse_replies"]))
    if "drop_think_samples" in d:
        cfg = replace(cfg, drop_think_samples=bool(d["drop_think_samples"]))
    if isinstance(d.get("think_markers"), list):
        cfg = replace(cfg, think_markers=[str(x) for x in d["think_markers"] if str(x)])
    if isinstance(d.get("refuse_extra_substrings"), list):
        cfg = replace(cfg, refuse_extra_substrings=[str(x) for x in d["refuse_extra_substrings"]])
    if "min_chars" in d:
        cfg = replace(cfg, min_chars=int(d["min_chars"]))
    if "max_chars" in d:
        mc = d["max_chars"]
        cfg = replace(cfg, max_chars=None if mc is None else int(mc))
    if "truncate" in d:
        cfg = replace(cfg, truncate=bool(d["truncate"]))
    if "max_non_printable_ratio" in d:
        cfg = replace(cfg, max_non_printable_ratio=float(d["max_non_printable_ratio"]))
    if "max_punctuation_ratio" in d:
        cfg = replace(cfg, max_punctuation_ratio=float(d["max_punctuation_ratio"]))
    if "allowed_langs" in d:
        al = d["allowed_langs"]
        if al is None:
            cfg = replace(cfg, allowed_langs=None)
        elif isinstance(al, list):
            cfg = replace(cfg, allowed_langs=[str(x) for x in al])
        else:
            raise TypeError("sft.allowed_langs 须为字符串列表或 null")
    if "lang_backend" in d:
        cfg = replace(cfg, lang_backend=str(d["lang_backend"]))
    if d.get("fasttext_model_path") is not None:
        cfg = replace(cfg, fasttext_model_path=str(d["fasttext_model_path"]))
    if "min_lang_confidence" in d:
        cfg = replace(cfg, min_lang_confidence=float(d["min_lang_confidence"]))
    if "exact_dedup" in d:
        cfg = replace(cfg, exact_dedup=bool(d["exact_dedup"]))
    if isinstance(d.get("exact"), dict):
        e = d["exact"]
        cfg = replace(cfg, exact=ExactDedupConfig(algorithm=str(e.get("algorithm", cfg.exact.algorithm))))
    if isinstance(d.get("near_dedup"), dict):
        n = d["near_dedup"]
        cfg = replace(
            cfg,
            near_dedup=NearDedupConfig(
                enabled=bool(n.get("enabled", cfg.near_dedup.enabled)),
                threshold=float(n.get("threshold", cfg.near_dedup.threshold)),
                num_perm=int(n.get("num_perm", cfg.near_dedup.num_perm)),
                shingle_size=int(n.get("shingle_size", cfg.near_dedup.shingle_size)),
            ),
        )
    if d.get("tokenizer_path_for_diagnostics") is not None:
        cfg = replace(cfg, tokenizer_path_for_diagnostics=str(d["tokenizer_path_for_diagnostics"]))
    if "max_rows_token_histogram" in d:
        cfg = replace(cfg, max_rows_token_histogram=int(d["max_rows_token_histogram"]))
    if "min_tokens" in d:
        v = d["min_tokens"]
        cfg = replace(cfg, min_tokens=None if v is None else int(v))
    if "max_tokens" in d:
        v = d["max_tokens"]
        cfg = replace(cfg, max_tokens=None if v is None else int(v))
    if d.get("stats_path") is not None:
        cfg = replace(cfg, stats_path=str(d["stats_path"]))
    if "run_diagnostics" in d:
        cfg = replace(cfg, run_diagnostics=bool(d["run_diagnostics"]))
    return cfg


def _pipeline_config_from_dict(d: dict[str, Any]) -> PreprocessPipelineConfig:
    cfg = PreprocessPipelineConfig()

    if "text_field" in d:
        cfg = replace(cfg, text_field=str(d["text_field"]))
    if "intra_doc_dedupe_paragraphs" in d:
        cfg = replace(cfg, intra_doc_dedupe_paragraphs=bool(d["intra_doc_dedupe_paragraphs"]))
    if "exact_dedup" in d:
        cfg = replace(cfg, exact_dedup=bool(d["exact_dedup"]))
    if "ppl_enable" in d:
        cfg = replace(cfg, ppl_enable=bool(d["ppl_enable"]))
    elif "ppl_filter" in d:
        # 兼容旧配置键名（建议迁移到 ppl_enable）。
        cfg = replace(cfg, ppl_enable=bool(d["ppl_filter"]))
    if "ppl_apply_percentile_filter" in d:
        cfg = replace(cfg, ppl_apply_percentile_filter=bool(d["ppl_apply_percentile_filter"]))
    if "ppl_low_percentile" in d:
        cfg = replace(cfg, ppl_low_percentile=float(d["ppl_low_percentile"]))
    if "ppl_high_percentile" in d:
        cfg = replace(cfg, ppl_high_percentile=float(d["ppl_high_percentile"]))
    if "ppl_keep_low_if_structured" in d:
        cfg = replace(cfg, ppl_keep_low_if_structured=bool(d["ppl_keep_low_if_structured"]))
    if "ppl_sort_ascending" in d:
        cfg = replace(cfg, ppl_sort_ascending=bool(d["ppl_sort_ascending"]))
    if "bert_mlm_log" in d:
        cfg = replace(cfg, bert_mlm_log=bool(d["bert_mlm_log"]))
    if "bert_mlm_sample_max" in d:
        cfg = replace(cfg, bert_mlm_sample_max=int(d["bert_mlm_sample_max"]))
    if "stats_path" in d and d["stats_path"] is not None:
        cfg = replace(cfg, stats_path=str(d["stats_path"]))

    if d.get("tokenizer_path_for_diagnostics") is not None:
        cfg = replace(cfg, tokenizer_path_for_diagnostics=str(d["tokenizer_path_for_diagnostics"]))
    if "run_diagnostics" in d:
        cfg = replace(cfg, run_diagnostics=bool(d["run_diagnostics"]))
    if "max_rows_token_histogram" in d:
        cfg = replace(cfg, max_rows_token_histogram=int(d["max_rows_token_histogram"]))

    if isinstance(d.get("basic"), dict):
        b = d["basic"]
        max_chars = cfg.basic.max_chars
        if "max_chars" in b:
            max_chars = b["max_chars"]
        allowed_langs = cfg.basic.allowed_langs
        if "allowed_langs" in b:
            al = b["allowed_langs"]
            if al is None:
                allowed_langs = None
            elif isinstance(al, list):
                allowed_langs = [str(x) for x in al]
            else:
                raise TypeError("basic.allowed_langs 须为字符串列表或 null")
        min_tok = cfg.basic.min_tokens
        if "min_tokens" in b:
            v = b["min_tokens"]
            min_tok = None if v is None else int(v)
        max_tok = cfg.basic.max_tokens
        if "max_tokens" in b:
            v = b["max_tokens"]
            max_tok = None if v is None else int(v)
        ft_path = cfg.basic.fasttext_model_path
        if "fasttext_model_path" in b:
            fmv = b["fasttext_model_path"]
            ft_path = None if fmv is None else str(fmv)
        cfg = replace(
            cfg,
            basic=BasicCleanConfig(
                min_chars=int(b.get("min_chars", cfg.basic.min_chars)),
                max_chars=max_chars,
                truncate=bool(b.get("truncate", cfg.basic.truncate)),
                max_non_printable_ratio=float(
                    b.get("max_non_printable_ratio", cfg.basic.max_non_printable_ratio)
                ),
                max_punctuation_ratio=float(
                    b.get("max_punctuation_ratio", cfg.basic.max_punctuation_ratio)
                ),
                allowed_langs=allowed_langs,
                lang_backend=str(b.get("lang_backend", cfg.basic.lang_backend)),
                fasttext_model_path=ft_path,
                min_lang_confidence=float(
                    b.get("min_lang_confidence", cfg.basic.min_lang_confidence)
                ),
                min_tokens=min_tok,
                max_tokens=max_tok,
            ),
        )

    if isinstance(d.get("exact"), dict):
        e = d["exact"]
        cfg = replace(cfg, exact=ExactDedupConfig(algorithm=str(e.get("algorithm", cfg.exact.algorithm))))

    if isinstance(d.get("near_dedup"), dict):
        n = d["near_dedup"]
        cfg = replace(
            cfg,
            near_dedup=NearDedupConfig(
                enabled=bool(n.get("enabled", cfg.near_dedup.enabled)),
                threshold=float(n.get("threshold", cfg.near_dedup.threshold)),
                num_perm=int(n.get("num_perm", cfg.near_dedup.num_perm)),
                shingle_size=int(n.get("shingle_size", cfg.near_dedup.shingle_size)),
            ),
        )

    if isinstance(d.get("ppl"), dict):
        g = d["ppl"]
        dev = cfg.ppl.device
        if "device" in g:
            dev = g["device"]
        cfg = replace(
            cfg,
            ppl=Gpt2PplConfig(
                model_name=str(g.get("model_name", cfg.ppl.model_name)),
                max_length=int(g.get("max_length", cfg.ppl.max_length)),
                device=dev,
            ),
        )

    if isinstance(d.get("bert_mlm"), dict):
        m = d["bert_mlm"]
        mdev = cfg.bert_mlm.device
        if "device" in m:
            mdev = m["device"]
        cfg = replace(
            cfg,
            bert_mlm=BertMlmNllConfig(
                model_name=str(m.get("model_name", cfg.bert_mlm.model_name)),
                max_length=int(m.get("max_length", cfg.bert_mlm.max_length)),
                max_masks=int(m.get("max_masks", cfg.bert_mlm.max_masks)),
                device=mdev,
            ),
        )

    if isinstance(d.get("topic"), dict):
        t = d["topic"]
        cfg = replace(
            cfg,
            topic=TopicAuditConfig(
                enabled=bool(t.get("enabled", cfg.topic.enabled)),
                max_docs=int(t.get("max_docs", cfg.topic.max_docs)),
                min_topic_size=int(t.get("min_topic_size", cfg.topic.min_topic_size)),
                output_dir=t.get("output_dir", cfg.topic.output_dir),
                embedding_model=str(t.get("embedding_model", cfg.topic.embedding_model)),
            ),
        )

    return cfg
