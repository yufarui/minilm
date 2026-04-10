"""JSONL 预处理流水线编排。"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.preprocess.basic_clean import BasicCleanConfig, apply_basic_text_quality, dedupe_consecutive_paragraphs
from src.preprocess.deduplicate import (
    ExactDedupConfig,
    NearDedupConfig,
    content_fingerprint,
    filter_by_mask,
    near_dedup_mask,
)
from src.preprocess.scoring import (
    BertMlmNllConfig,
    Gpt2PplConfig,
    bert_mlm_mean_nll,
    gpt2_perplexities,
    percentile_bounds,
    ppl_keep_mask,
)
from src.preprocess.stats_types import PreprocessPipelineStats, TopicAuditStats
from src.preprocess.text_quality.length import equal_width_histogram
from src.preprocess.text_quality.tokens import (
    TokenBoundsConfig,
    histogram_from_lengths,
    needs_tokenizer as token_bounds_configured,
    top_token_entries,
)
from src.preprocess.topic_audit import TopicAuditConfig, run_topic_audit
from src.ref_model import get_auto_tokenizer_local
from src.util.path_util import resolve_under_project

logger = logging.getLogger(__name__)


@dataclass
class PreprocessPipelineConfig:
    """预训练文本预处理总开关：清洗、去重、打分和诊断等阶段配置。"""
    # JSONL 中的文本字段名。
    text_field: str = "text"
    # 基础文本质量清洗配置。
    basic: BasicCleanConfig = field(default_factory=BasicCleanConfig)
    # 是否在单文档内去重连续重复段落。
    intra_doc_dedupe_paragraphs: bool = True
    # 是否启用精确去重。
    exact_dedup: bool = True
    # 精确去重参数（哈希算法等）。
    exact: ExactDedupConfig = field(default_factory=ExactDedupConfig)
    # 近似去重参数（MinHash/LSH 等）。
    near_dedup: NearDedupConfig = field(default_factory=NearDedupConfig)
    # 是否计算参考模型 PPL。
    ppl_enable: bool = False
    """为 True 时用参考模型计算每条文档 PPL，并写入 ``_ppl`` 与分位阈值统计。"""
    # 是否按 PPL 分位阈值真正过滤样本。
    ppl_apply_percentile_filter: bool = False
    """为 True 时按分位数实际删文档；为 False 时只观测分布（默认）。"""
    # PPL 计算参数（模型、批大小等）。
    ppl: Gpt2PplConfig = field(default_factory=Gpt2PplConfig)
    # PPL 过滤下分位点（百分位数）。
    ppl_low_percentile: float = 1.0
    # PPL 过滤上分位点（百分位数）。
    ppl_high_percentile: float = 95.0
    # 是否保留“低 PPL 但结构化明显”的文本。
    ppl_keep_low_if_structured: bool = True
    # 是否按 PPL 从低到高重排输出。
    ppl_sort_ascending: bool = False
    """为 True 时，在 PPL 计算后将输出样本按 ``(ppl, length)`` 从低到高重排。"""
    # 是否记录 BERT MLM NLL 指标（仅统计，不过滤）。
    bert_mlm_log: bool = False
    # BERT MLM 评分参数。
    bert_mlm: BertMlmNllConfig = field(default_factory=BertMlmNllConfig)
    # BERT MLM 打分最多采样文档数。
    bert_mlm_sample_max: int = 256
    # 主题审计配置。
    topic: TopicAuditConfig = field(default_factory=TopicAuditConfig)
    # 统计信息输出路径（None 表示不落盘）。
    stats_path: str | None = None
    # 诊断与 token 上下界过滤用 tokenizer 路径。
    tokenizer_path_for_diagnostics: str | None = None
    """提供且启用诊断时用于 token 长度分布、与 ``basic`` 中 token 上下界过滤。"""
    # 是否启用诊断统计。
    run_diagnostics: bool = False
    """为 True 时在 ``stats.diagnostics`` 中写入文本质量剔除计数与 token 分布（需 tokenizer 路径）。"""
    # 诊断阶段最多统计多少行 token 直方图。
    max_rows_token_histogram: int = 2000


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class PreprocessPipeline:
    def __init__(self, cfg: PreprocessPipelineConfig) -> None:
        self.cfg = cfg

    def run(self, input_path: str | Path, output_path: str | Path) -> PreprocessPipelineStats:
        inp = Path(input_path)
        out = Path(output_path)
        stats = PreprocessPipelineStats()

        rows_in: list[dict[str, Any]] = []
        for obj in _iter_jsonl(inp):
            stats.input_lines += 1
            rows_in.append(obj)

        texts_raw: list[str] = []
        meta: list[dict[str, Any]] = []
        for obj in rows_in:
            tf = self.cfg.text_field
            raw = obj.get(tf, "")
            texts_raw.append(str(raw) if raw is not None else "")
            rest = {k: v for k, v in obj.items() if k != tf}
            meta.append(rest)

        tok = None
        need_tok = token_bounds_configured(
            TokenBoundsConfig(
                min_tokens=self.cfg.basic.min_tokens,
                max_tokens=self.cfg.basic.max_tokens,
            )
        )
        if need_tok or (self.cfg.run_diagnostics and self.cfg.tokenizer_path_for_diagnostics):
            if not self.cfg.tokenizer_path_for_diagnostics:
                if need_tok:
                    raise ValueError("basic 中设置了 min_tokens/max_tokens 时必须配置 tokenizer_path_for_diagnostics")
            else:
                try:
                    tok = get_auto_tokenizer_local(self.cfg.tokenizer_path_for_diagnostics, trust_remote_code=True)
                except Exception as e:
                    if need_tok:
                        raise
                    logger.warning("无法加载 tokenizer，跳过预训练 token 诊断：%s", e)

        reject_counts: Counter[str] = Counter()
        cleaned: list[str | None] = []
        for t in texts_raw:
            res = apply_basic_text_quality(t, self.cfg.basic, tokenizer=tok)
            if res.text is None:
                if res.reject:
                    reject_counts[res.reject] += 1
                cleaned.append(None)
                continue
            u = res.text
            if self.cfg.intra_doc_dedupe_paragraphs:
                u = dedupe_consecutive_paragraphs(u)
            cleaned.append(u)

        kept_idx = [i for i, t in enumerate(cleaned) if t is not None]
        stats.after_basic_clean = len(kept_idx)
        texts = [cleaned[i] for i in kept_idx]
        metas = [meta[i] for i in kept_idx]

        diag: dict[str, Any] = {"text_quality_rejects": dict(reject_counts)}
        token_lens: list[float] = []
        token_counter: Counter[int] = Counter()
        diag_rows = 0
        if self.cfg.run_diagnostics and tok is not None:
            for t in texts:
                if diag_rows >= self.cfg.max_rows_token_histogram:
                    break
                ids = tok.encode(t, add_special_tokens=False)
                token_lens.append(float(len(ids)))
                for tid in ids:
                    token_counter[tid] += 1
                diag_rows += 1
            if token_lens:
                h = histogram_from_lengths(token_lens, bins=20)
                diag["doc_token_len_hist_edges"] = h["edges"]
                diag["doc_token_len_hist_counts"] = h["counts"]
            if token_counter:
                diag["token_id_top"] = top_token_entries(token_counter, tok, k=40)
        stats.diagnostics = diag

        if self.cfg.exact_dedup:
            seen_hashes: set[str] = set()
            uniq_texts: list[str] = []
            uniq_meta: list[dict[str, Any]] = []
            for t, m in zip(texts, metas, strict=True):
                h = content_fingerprint(t, algorithm=self.cfg.exact.algorithm)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                uniq_texts.append(t)
                uniq_meta.append(m)
            texts, metas = uniq_texts, uniq_meta

        stats.after_exact_dedup = len(texts)

        if self.cfg.near_dedup.enabled:
            mask = near_dedup_mask(texts, self.cfg.near_dedup)
            texts = filter_by_mask(texts, mask)
            metas = filter_by_mask(metas, mask)
        stats.after_near_dedup = len(texts)

        ppls: list[float] | None = None
        if self.cfg.ppl_enable:
            logger.info("Computing reference PPL for %s documents …", len(texts))
            ppls = gpt2_perplexities(texts, self.cfg.ppl, progress_every=200)
            lo, hi = percentile_bounds(
                ppls, self.cfg.ppl_low_percentile, self.cfg.ppl_high_percentile
            )
            stats.ppl_threshold_low = lo if lo == lo else None
            stats.ppl_threshold_high = hi if hi == hi else None
            keep_p = ppl_keep_mask(
                texts,
                ppls,
                self.cfg.ppl_low_percentile,
                self.cfg.ppl_high_percentile,
                keep_low_if_structured=self.cfg.ppl_keep_low_if_structured,
                thresholds=(lo, hi),
            )
            if self.cfg.ppl_apply_percentile_filter:
                texts = filter_by_mask(texts, keep_p)
                metas = filter_by_mask(metas, keep_p)
                ppls = filter_by_mask(ppls, keep_p)
            else:
                stats.ppl_would_remove_count = sum(1 for k in keep_p if not k)
            if self.cfg.ppl_sort_ascending and ppls is not None:
                order = sorted(
                    range(len(texts)),
                    key=lambda i: (
                        not (ppls[i] == ppls[i]),  # NaN 最后
                        ppls[i] if (ppls[i] == ppls[i]) else float("inf"),
                        len(texts[i]),
                        i,
                    ),
                )
                texts = [texts[i] for i in order]
                metas = [metas[i] for i in order]
                ppls = [ppls[i] for i in order]
        stats.after_ppl_filter = len(texts)

        if self.cfg.bert_mlm_log and texts:
            sample = texts[: self.cfg.bert_mlm_sample_max]
            logger.info("BERT MLM NLL on %s docs …", len(sample))
            bert_scores = bert_mlm_mean_nll(sample, self.cfg.bert_mlm)
            if bert_scores:
                stats.bert_mlm_mean_nll = float(sum(bert_scores) / len(bert_scores))

        if self.cfg.topic.enabled:
            if texts:
                stats.topic_audit = run_topic_audit(texts, self.cfg.topic)
            else:
                stats.topic_audit = TopicAuditStats(ran=False, error="empty_corpus")
        else:
            stats.topic_audit = TopicAuditStats(skipped=True)

        out_rows: list[dict[str, Any]] = []
        for i, t in enumerate(texts):
            row = dict(metas[i])
            row[self.cfg.text_field] = t
            if ppls is not None:
                row["_ppl"] = float(ppls[i])
            out_rows.append(row)

        lengths = [float(len(t)) for t in texts]
        lbins, lcnt = equal_width_histogram(lengths, bins=30)
        stats.doc_length_hist_bins = lbins
        stats.doc_length_hist_counts = lcnt
        if ppls is not None:
            pbins, pcnt = equal_width_histogram(ppls, bins=30)
            stats.ppl_hist_bins = pbins
            stats.ppl_hist_counts = pcnt

        stats.output_lines = len(out_rows)
        _write_jsonl(out, out_rows)
        logger.info("Wrote %s lines → %s", len(out_rows), out)

        if self.cfg.stats_path:
            sp = resolve_under_project(self.cfg.stats_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            with sp.open("w", encoding="utf-8") as sf:
                json.dump(stats.to_json_dict(), sf, ensure_ascii=False, indent=2)

        return stats
