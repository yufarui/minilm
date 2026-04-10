"""SFT 对话 JSONL 预处理：诊断、清洗、与预训练共用的去重/文本质量策略。"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.config.logging_config import setup_logging
from src.preprocess.basic_clean import BasicCleanConfig, apply_basic_text_quality
from src.preprocess.deduplicate import (
    ExactDedupConfig,
    NearDedupConfig,
    content_fingerprint,
    filter_by_mask,
    near_dedup_mask,
)
from src.preprocess.shared_text import (
    looks_like_refuse_reply,
    normalize_special_markers,
)
from src.preprocess.sft_conversation import (
    assistant_contents,
    conversation_concat_text,
    count_turns,
    normalize_messages_tool_calls,
    tool_calls_json_length,
    validate_role_chain,
)
from src.preprocess.stats_types import SftPreprocessStats
from src.preprocess.text_quality.length import equal_width_histogram
from src.preprocess.text_quality.pipeline import (
    REJECT_LANGUAGE,
    REJECT_LENGTH,
    REJECT_NON_PRINTABLE,
    REJECT_PUNCTUATION,
    REJECT_TOKENS,
)
from src.preprocess.text_quality.tokens import (
    TokenBoundsConfig,
    needs_tokenizer as token_bounds_configured,
    top_token_entries,
)
from src.ref_model import get_auto_tokenizer_local
from src.util.path_util import resolve_under_project

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        for r in rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")


def _apply_markers_to_messages(messages: list[dict[str, Any]], reps: dict[str, str]) -> list[dict[str, Any]]:
    if not reps:
        return messages
    out = []
    for m in messages:
        mm = dict(m) if isinstance(m, dict) else m
        if isinstance(mm, dict) and isinstance(mm.get("content"), str):
            mm = dict(mm)
            mm["content"] = normalize_special_markers(mm["content"], reps)
        out.append(mm)
    return out


@dataclass
class SftPipelineConfig:
    """SFT 对话预处理配置：角色校验、修复、清洗、去重与诊断。"""
    # 对话列表字段名。
    conversations_field: str = "conversations"
    # 是否严格校验角色顺序（通常 user/assistant 交替）。
    strict_role_order: bool = False
    # 是否尝试修复不规范 tool_calls JSON 字符串。
    repair_tool_calls: bool = True
    # 特殊标记替换映射（如 <think> -> <reason>）。
    normalize_markers: dict[str, str] = field(default_factory=dict)
    # 是否过滤拒答类 assistant 回复。
    filter_refuse_replies: bool = True
    # 额外拒答关键词（在默认规则之外）。
    refuse_extra_substrings: list[str] = field(default_factory=list)
    # 基础清洗最小字符数。
    min_chars: int = 20
    # 基础清洗最大字符数（None 表示不设上限）。
    max_chars: int | None = 200_000
    # 超长文本是否截断到上限。
    truncate: bool = True
    # 非可打印字符占比上限。
    max_non_printable_ratio: float = 0.05
    # 标点占比上限。
    max_punctuation_ratio: float = 0.55
    # 允许语种列表（None 用默认语种；[] 表示不过滤）。
    allowed_langs: list[str] | None = None
    """``None``：``zh-cn``+``en``；``[]``：不过滤语言。"""
    # 语种识别后端。
    lang_backend: str = "fasttext"
    """仅支持 ``fasttext``。"""
    # fastText 模型路径（为空时使用默认加载逻辑）。
    fasttext_model_path: str | None = None
    # 语种识别最小置信度阈值。
    min_lang_confidence: float = 0.0
    """语种预测置信度下限（当前仅 fasttext 生效）。"""
    # 最小 token 数（None 表示不限制）。
    min_tokens: int | None = None
    # 最大 token 数（None 表示不限制）。
    max_tokens: int | None = None
    # 是否启用精确去重。
    exact_dedup: bool = True
    # 精确去重参数。
    exact: ExactDedupConfig = field(default_factory=ExactDedupConfig)
    # 近似去重参数。
    near_dedup: NearDedupConfig = field(default_factory=NearDedupConfig)
    # 诊断与 token 上下界过滤用 tokenizer 路径。
    tokenizer_path_for_diagnostics: str | None = None
    # 诊断阶段最多统计多少行 token 直方图。
    max_rows_token_histogram: int = 2000
    # 统计信息输出路径（None 表示不落盘）。
    stats_path: str | None = None
    # 是否启用诊断统计。
    run_diagnostics: bool = True
    """为 True 时在 stats.diagnostics 中写入长度/轮数/token 分布等。"""


def _sft_basic_config(c: SftPipelineConfig) -> BasicCleanConfig:
    return BasicCleanConfig(
        min_chars=c.min_chars,
        max_chars=c.max_chars,
        truncate=c.truncate,
        max_non_printable_ratio=c.max_non_printable_ratio,
        max_punctuation_ratio=c.max_punctuation_ratio,
        allowed_langs=c.allowed_langs,
        lang_backend=c.lang_backend,
        fasttext_model_path=c.fasttext_model_path,
        min_lang_confidence=c.min_lang_confidence,
        min_tokens=c.min_tokens,
        max_tokens=c.max_tokens,
    )


def _bump_reject(stats: SftPreprocessStats, reject: str | None) -> None:
    if reject == REJECT_LENGTH:
        stats.skipped_length += 1
    elif reject == REJECT_NON_PRINTABLE:
        stats.skipped_non_printable += 1
    elif reject == REJECT_PUNCTUATION:
        stats.skipped_punctuation += 1
    elif reject == REJECT_LANGUAGE:
        stats.skipped_language += 1
    elif reject == REJECT_TOKENS:
        stats.skipped_tokens += 1


class SftPreprocessPipeline:
    def __init__(self, cfg: SftPipelineConfig) -> None:
        self.cfg = cfg

    def run(self, input_path: str | Path, output_path: str | Path) -> SftPreprocessStats:
        setup_logging()
        inp, out = Path(input_path), Path(output_path)
        stats = SftPreprocessStats()
        cf = self.cfg.conversations_field
        basic_cfg = _sft_basic_config(self.cfg)

        rows_in: list[dict[str, Any]] = []
        for obj in _iter_jsonl(inp):
            stats.input_lines += 1
            rows_in.append(obj)

        need_tok = token_bounds_configured(
            TokenBoundsConfig(min_tokens=self.cfg.min_tokens, max_tokens=self.cfg.max_tokens)
        )
        tok = None
        if need_tok or (self.cfg.run_diagnostics and self.cfg.tokenizer_path_for_diagnostics):
            if not self.cfg.tokenizer_path_for_diagnostics:
                if need_tok:
                    raise ValueError("设置了 min_tokens/max_tokens 时必须配置 tokenizer_path_for_diagnostics")
            else:
                try:
                    tok = get_auto_tokenizer_local(self.cfg.tokenizer_path_for_diagnostics, trust_remote_code=True)
                except Exception as e:
                    if need_tok:
                        raise
                    logger.warning("无法加载诊断用 tokenizer，跳过 token 直方图：%s", e)

        user_lens: list[float] = []
        asst_lens: list[float] = []
        tool_json_lens: list[float] = []
        turns_list: list[float] = []
        token_counter: Counter[int] = Counter()
        diag_rows = 0

        cleaned_rows: list[dict[str, Any]] = []
        texts_for_near: list[str] = []

        for obj in rows_in:
            conv = obj.get(cf)
            if not isinstance(conv, list) or not conv:
                stats.skipped_empty_conversations += 1
                continue

            messages = copy.deepcopy(conv)
            if self.cfg.normalize_markers:
                messages = _apply_markers_to_messages(messages, self.cfg.normalize_markers)
                stats.markers_normalized_rows += 1

            if self.cfg.repair_tool_calls:
                messages, nrep = normalize_messages_tool_calls(messages)
                stats.tool_calls_repaired += nrep

            if self.cfg.strict_role_order:
                ok, reason = validate_role_chain(messages)
                if not ok:
                    stats.skipped_role_order += 1
                    if len(stats.role_violation_examples) < 8:
                        stats.role_violation_examples.append(reason or "role")
                    continue

            if self.cfg.filter_refuse_replies:
                bad = False
                for ac in assistant_contents(messages):
                    if looks_like_refuse_reply(ac, self.cfg.refuse_extra_substrings):
                        bad = True
                        break
                if bad:
                    stats.skipped_refuse_reply += 1
                    continue

            flat = conversation_concat_text(messages)
            res = apply_basic_text_quality(flat, basic_cfg, tokenizer=tok)
            if res.text is None:
                _bump_reject(stats, res.reject)
                continue

            tnorm = res.text
            new_obj = {k: v for k, v in obj.items() if k != cf}
            new_obj[cf] = messages
            cleaned_rows.append(new_obj)
            texts_for_near.append(tnorm)

            if self.cfg.run_diagnostics and tok is not None and diag_rows < self.cfg.max_rows_token_histogram:
                turns_list.append(float(count_turns(messages)))
                tool_json_lens.append(float(tool_calls_json_length(messages)))
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role")
                    c = m.get("content")
                    if not isinstance(c, str) or not c.strip():
                        continue
                    ids = tok.encode(c, add_special_tokens=False)
                    if role == "user":
                        user_lens.append(float(len(ids)))
                    elif role == "assistant":
                        asst_lens.append(float(len(ids)))
                    for tid in ids:
                        token_counter[tid] += 1
                diag_rows += 1

        after_clean = len(cleaned_rows)
        stats.after_exact_dedup = after_clean
        stats.after_near_dedup = after_clean

        if self.cfg.exact_dedup:
            seen: set[str] = set()
            uniq_rows: list[dict[str, Any]] = []
            uniq_texts: list[str] = []
            for row, tx in zip(cleaned_rows, texts_for_near, strict=True):
                key = json.dumps(row.get(cf, []), ensure_ascii=False, sort_keys=True)
                h = content_fingerprint(key, algorithm=self.cfg.exact.algorithm)
                if h in seen:
                    continue
                seen.add(h)
                uniq_rows.append(row)
                uniq_texts.append(tx)
            cleaned_rows, texts_for_near = uniq_rows, uniq_texts
            stats.after_exact_dedup = len(cleaned_rows)

        if self.cfg.near_dedup.enabled and cleaned_rows:
            mask = near_dedup_mask(texts_for_near, self.cfg.near_dedup)
            cleaned_rows = filter_by_mask(cleaned_rows, mask)
            texts_for_near = filter_by_mask(texts_for_near, mask)
            stats.after_near_dedup = len(cleaned_rows)

        stats.output_lines = len(cleaned_rows)
        if turns_list:
            stats.mean_turns = sum(turns_list) / len(turns_list)

        if self.cfg.run_diagnostics:
            diag: dict[str, Any] = {}
            if user_lens:
                e, c = equal_width_histogram(user_lens, bins=20)
                diag["user_token_len_hist_edges"] = e
                diag["user_token_len_hist_counts"] = c
            if asst_lens:
                e, c = equal_width_histogram(asst_lens, bins=20)
                diag["assistant_token_len_hist_edges"] = e
                diag["assistant_token_len_hist_counts"] = c
            if tool_json_lens:
                e, c = equal_width_histogram(tool_json_lens, bins=20)
                diag["tool_calls_char_len_hist_edges"] = e
                diag["tool_calls_char_len_hist_counts"] = c
            if turns_list:
                e, c = equal_width_histogram(turns_list, bins=min(20, max(5, int(max(turns_list)) + 1)))
                diag["turns_hist_edges"] = e
                diag["turns_hist_counts"] = c
            if token_counter and tok is not None:
                diag["token_id_top"] = top_token_entries(token_counter, tok, k=40)
            stats.diagnostics = diag

        _write_jsonl(out, cleaned_rows)
        logger.info("SFT preprocess wrote %s lines → %s", len(cleaned_rows), out)

        if self.cfg.stats_path:
            sp = resolve_under_project(self.cfg.stats_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            with sp.open("w", encoding="utf-8") as sf:
                json.dump(stats.to_json_dict(), sf, ensure_ascii=False, indent=2)

        return stats
