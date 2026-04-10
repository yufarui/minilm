"""可选：BERTopic 领域/主题分布审计（需安装 bertopic 及依赖）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocess.stats_types import TopicAuditStats


@dataclass
class TopicAuditConfig:
    enabled: bool = False
    max_docs: int = 10_000
    min_topic_size: int = 10
    output_dir: str | None = None
    embedding_model: str = "all-MiniLM-L6-v2"


def run_topic_audit(texts: list[str], cfg: TopicAuditConfig) -> TopicAuditStats:
    if not cfg.enabled:
        return TopicAuditStats(skipped=True)
    try:
        from bertopic import BERTopic
    except ImportError as e:
        raise ImportError(
            "主题审计需要 bertopic 与 sentence-transformers："
            "pip install minilm[preprocess-full] 或手动安装 bertopic sentence-transformers"
        ) from e

    sample = texts[: cfg.max_docs]
    if not sample:
        return TopicAuditStats(ran=False, error="empty")

    from src.ref_model import get_sentence_transformer

    embedding_model = get_sentence_transformer(cfg.embedding_model)
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=cfg.min_topic_size)
    topic_model.fit_transform(sample)
    info = topic_model.get_topic_info()

    st = TopicAuditStats(
        ran=True,
        n_docs=len(sample),
        n_topics=int(info.shape[0]) - 1,
        topic_info_rows=int(info.shape[0]),
    )

    if cfg.output_dir:
        p = Path(cfg.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        info.to_csv(p / "topic_info.csv", index=False)
        try:
            fig = topic_model.visualize_barchart(top_n_topics=min(20, max(len(info), 1)))
            fig.write_html(str(p / "topics_barchart.html"))
        except Exception as e:
            st.visualize_error = str(e)
        st.output_dir = str(p.resolve())

    return st
